"""Automatically generate target information from clini table

# The `barspoon-targets 3.0` File Format

A barspoon target file is a [TOML][1] file with the following entries:

  - A `version` key mapping to a version string `"barspoon-targets <V>"`, where
    `<V>` is a [PEP-440 version string][2] compatible with `2.0`.
  - A `targets` table, the keys of which are target labels (as found in the
    clinical table) and the values specify exactly one of the following:
     1. A categorical target label, marked by the presence of a `type =
        "category"` key-value pair.
     2. A target label to quantize, marked by the presence of a `type =
        "quantize"` key-value pair.
     3. A target format defined in in a later version of barspoon targets, maked
        by the presence of a `type = X` key-value pair, where `X` is a string
        which is not `"category"` or `"quantize"`.
    A definition of these entries can be found below.

[1]: https://toml.io "Tom's Obvious Minimal Language"
[2]: https://peps.python.org/pep-0440/
    "PEP 440 - Version Identification and Dependency Specification"

## Targets

### Categorical Target Label

A categorical target is a target table with a key-value pair `type =
"category"`.  `categories` contains a list of lists of literal strings.  Each
list of strings will be treated as one category, with all literal strings within
that list being treated as one representative for that category.  This allows
the user to easily group related classes into one large class (i.e. `"True",
"1", "Yes"` could all be unified into the same category).

#### Category Weights

It is possible to assign a weight to each category, to e.g. weigh rarer classes
more heavily.  The weights are stored in a table `targets.LABEL.class_weights`,
whose keys is the first representative of each category, and the values of which
is the weight of the category as a floating point number.

### Target Label to Quantize

If a target has key-value pair `type = "quantize"`, it is interpreted as a
continuous target which has to be quantized.  `thresholds` has to be a list of
floating point numbers [t_0, t_n], n > 1 containing the thresholds of the bins
to quantize the values into.  A categorical target will be quantized into bins

```asciimath
b_0 = [t_0; t_1], b_1 = (t_1; b_2], ... b_(n-1) = (t_(n-1); t_n]
```

The bins will be treated as categories with names
`f"[{t_0:+1.2e};{t_1:+1.2e}]"` for the first bin and
`f"({t_i:+1.2e};{t_(i+1):+1.2e}]"` for all other bins

To avoid confusion, we recommend to also format the `thresholds` list the same
way.

The bins can also be weighted. See _Categorical Target Label: Category Weights_
for details.

  > Experience has shown that many labels contain non-negative values with a
  > disproportionate amount (more than n_samples/n_bins) of zeroes. We thus
  > decided to make the _right_ side of each bin inclusive, as the bin (-A,0]
  > then naturally includes those zero values.

## Additional Inputs

It is also possible to mark some labels as additional inputs.  These are stored
in the `additional_inputs` table instead of the `targets` table.  Apart from
that, additional input specifications follow the same rules as the target
specifications described above.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn.functional as F
from packaging.specifiers import Specifier

from barspoon.utils import read_table

__all__ = ["encode", "decode_targets", "Categorical"]


class Categorical(NamedTuple):
    categories: List[str]
    encoded: torch.Tensor
    weight: torch.Tensor


def encode(
    clini_df: pd.DataFrame,
    *,
    # From targets toml
    version: str = "barspoon-targets 3.0",
    targets: Mapping[str, Any],
    additional_inputs: Optional[Mapping[str, Any]] = None,
    **ignored,
) -> Tuple[Dict[str, Categorical], Dict[str, Categorical]]:
    """Encodes the information in a clini table into a tensor"""
    # Make sure target file has the right version
    name, version = version.split(" ")
    if not (
        name == "barspoon-targets" and (spec := Specifier("~=3.0")).contains(version)
    ):
        raise ValueError(
            f"incompatible target file: expected barspoon-targets{spec}, found `{name} {version}`"
        )

    if ignored:
        logging.warn(f"ignored {ignored}")

    encoded_targets = {}
    for target_label, target_spec in (targets or {}).items():
        if target_label not in clini_df:
            logging.warning(f"no clinical data for {target_label}.  Skipping...")
            continue
        encoded_targets[target_label] = _encode(
            clini_df=clini_df, label_spec=target_spec, label=target_label
        )

    encoded_additionals = {}
    for additional_label, additional_spec in (additional_inputs or {}).items():
        if additional_label not in clini_df:
            logging.warning(f"no clinical data for {additional_label}.  Skipping...")
            continue
        encoded_additionals[additional_label] = _encode(
            clini_df=clini_df, label_spec=additional_spec, label=additional_label
        )

    return encoded_targets, encoded_additionals


def _encode(*, clini_df, label_spec, label) -> Categorical:
    if label_spec["type"] == "category":
        representatives, encoded, weight = encode_category(
            clini_df=clini_df, target_label=label, **label_spec
        )
        return Categorical(categories=representatives, encoded=encoded, weight=weight)

    elif label_spec["type"] == "quantize":
        representatives, encoded, weight = encode_quantize(
            clini_df=clini_df, target_label=label, **label_spec
        )
        return Categorical(categories=representatives, encoded=encoded, weight=weight)

    else:
        logging.warn(f"ignoring unrecognized target type {label}")


def encode_category(
    *,
    clini_df: pd.DataFrame,
    target_label: str,
    type: str,
    categories: Sequence[List[str]],
    class_weights: Optional[Dict[str, float]] = None,
    **ignored,
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    assert type == "category"
    # Map each category to its index
    category_map = {member: idx for idx, cat in enumerate(categories) for member in cat}

    # Map each item to it's category's index, mapping nans to num_classes+1
    # This way we can easily discard the NaN column later
    indexes = clini_df[target_label].map(lambda c: category_map.get(c, len(categories)))
    indexes = torch.tensor(indexes.values)

    # Discard nan column
    one_hot = F.one_hot(indexes, num_classes=len(categories) + 1)[:, :-1]

    # Class weights
    if class_weights is not None:
        weight = torch.tensor([class_weights[c[0]] for c in categories])
    else:
        # No class weights given; use normalized inverse frequency
        counts = one_hot.sum(dim=0)
        weight = (w := (counts.sum() / counts)) / w.sum()

    # Warn user of unused labels
    if ignored:
        logging.warn(f"ignored labels in target {target_label}: {ignored}")

    return [c[0] for c in categories], one_hot, weight


def encode_quantize(
    *,
    clini_df: pd.DataFrame,
    target_label: str,
    type: str,
    thresholds: List[float],
    class_weights: Optional[Dict[str, float]] = None,
    **ignored,
) -> Tuple[torch.Tensor, torch.Tensor, npt.NDArray[np.str_]]:
    assert type == "quantize"
    # Warn user of unused labels
    if ignored:
        logging.warn(f"ignored labels in target {target_label}: {ignored}")

    n_bins = len(thresholds) - 1
    numeric_vals = torch.tensor(pd.to_numeric(clini_df[target_label]).values).reshape(
        -1, 1
    )

    # Map each value to a class index as follows:
    #  1. If the value is NaN or less than the left-most threshold, use class
    #     index 0
    #  2. If it is between the left-most and the right-most threshold, set it to
    #     the bin number (starting from 1)
    #  3. If it is larger than the right-most threshold, set it to N_bins + 1
    bin_index = (
        (numeric_vals > torch.tensor(thresholds).reshape(1, -1)).count_nonzero(1)
        # For the first bucket, we have to include the lower threshold
        + (numeric_vals.reshape(-1) == thresholds[0])
    )
    # One hot encode and discard nan columns (first and last col)
    one_hot = F.one_hot(bin_index, num_classes=n_bins + 2)[:, 1:-1]

    # Class weights
    categories = [
        f"[{thresholds[0]:+1.2e};{thresholds[1]:+1.2e}]",
        *(
            f"({l:+1.2e};{u:+1.2e}]"
            for l, u in zip(thresholds[1:-1], thresholds[2:], strict=True)
        ),
    ]

    if class_weights is not None:
        weight = torch.tensor([class_weights[c] for c in categories])
    else:
        # No class weights given; use normalized inverse frequency
        counts = one_hot.sum(0)
        weight = (w := (np.divide(counts.sum(), counts, where=counts > 0))) / w.sum()

    return np.array(categories), one_hot, weight


def decode_targets(
    encoded: torch.Tensor,
    *,
    target_labels: Sequence[str],
    targets: Dict[str, Any],
    version: str = "barspoon-targets 3.0",
    **ignored,
) -> List[np.array]:
    name, version = version.split(" ")
    if not (
        name == "barspoon-targets" and (spec := Specifier("~=3.0")).contains(version)
    ):
        raise ValueError(f"model not compatible with barspoon-targets {spec}", version)

    # Warn user of unused labels
    if ignored:
        logging.warn(f"ignored parameters: {ignored}")

    decoded_targets = []
    curr_col = 0
    for target_label in target_labels:
        info = targets[target_label]

        if info["type"] == "category":
            categories = info["categories"]
            # Add another column which is one iff all the other values are zero
            encoded_target = encoded[:, curr_col : curr_col + len(categories)]
            is_none = ~encoded_target.any(dim=1).view(-1, 1)
            encoded_target = torch.cat([encoded_target, is_none], dim=1)

            # Decode to class labels
            representatives = np.array([c[0] for c in categories] + [None])
            category_index = encoded_target.argmax(dim=1)
            decoded = representatives[category_index]
            decoded_targets.append(decoded)

            curr_col += len(categories)

        elif info["type"] == "quantize":
            thresholds = info["thresholds"]
            n_bins = len(thresholds) - 1
            encoded_target = encoded[:, curr_col : curr_col + n_bins]
            is_none = ~encoded_target.any(dim=1).view(-1, 1)
            encoded_target = torch.cat([encoded_target, is_none], dim=1)

            bin_edges = [-np.inf, *thresholds, np.inf]
            representatives = np.array(
                [
                    f"[{l:+1.2e};{u:+1.2e})"
                    for l, u in zip(bin_edges[:-1], bin_edges[1:])
                ]
            )
            decoded = representatives[encoded_target.argmax(dim=1)]

            decoded_targets.append(decoded)

            curr_col += n_bins

        else:
            raise ValueError(f"cannot decode {target_label} of type {info['type']}")

    return decoded_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output-file",
        metavar="PATH",
        type=lambda p: sys.stdout if p == "--" else open(p, "w"),
        default=sys.stdout,
        help="File to output classification info into, or `--` for stdout",
    )

    parser.add_argument(
        "-c",
        "--clini-table",
        metavar="PATH",
        dest="clini_tables",
        type=Path,
        required=True,
        action="append",
        help="Path to the clinical table. Can be specified multiple times",
    )

    parser.add_argument(
        "--category",
        metavar="LABEL",
        dest="categorical_labels",
        type=str,
        default=[],
        action="append",
        help="Label to categorize. Can be specified multiple times",
    )
    parser.add_argument(
        "--category-min-count",
        metavar="N",
        type=int,
        default=32,
        help="Minimal support of for a category to be included",
    )

    parser.add_argument(
        "--quantize",
        nargs=2,
        metavar=("LABEL", "N"),
        type=str,
        default=[],
        action="append",
        help="Label to binarize. Can be specified multiple times",
    )
    args = parser.parse_args()
    args.quantize = [(label, int(n)) for label, n in args.quantize]

    outtoml = args.output_file
    clini_df = pd.concat([read_table(c) for c in args.clini_tables])

    # Artifact name and version
    outtoml.write('version = "barspoon-targets 3.0"\n\n')

    # Translation table to escape basic strings in TOML
    escape_table = str.maketrans(
        {
            **{chr(i): f"\\u{i:04X}" for i in range(0x0, 0x8)},
            "\b": "\\b",  # u0008
            "\t": "\\t",  # u0009
            "\n": "\\n",  # u000A
            "\x0b": "\\u000B",
            "\f": "\\f",  # u000C
            "\r": "\\r",  # u000D
            "\x0e": "\\u000E",
            "\x0f": "\\u000F",
            "\x7f": "\\u007F",
        }
    )

    # "True" categorical variables
    for target_label in args.categorical_labels:
        counts = clini_df[target_label].value_counts()
        # Comment out section if there are fewer than two well-populated classes
        prefix = "#" if (counts >= args.category_min_count).sum() <= 1 else ""

        outtoml.write(f'{prefix}[targets."{target_label.translate(escape_table)}"]\n')
        outtoml.write(f'{prefix}type = "category"\n')

        # List all categories with little-populated categories being commented
        # out
        outtoml.write(f"{prefix}categories = [\n")
        for cat, n in sorted(counts.items()):
            if n < args.category_min_count:
                outtoml.write("#")
            outtoml.write(
                f'{prefix}\t["{cat.translate(escape_table)}"],\t# count = {n}\n'
            )
        outtoml.write(f"{prefix}]\n")

        # Calculate weights of well-populated classes
        # inverse to their frequency of occurrence
        well_supported_counts = counts[counts >= args.category_min_count]
        pos_weights = well_supported_counts.sum() / well_supported_counts
        pos_weights /= pos_weights.sum()
        outtoml.write(
            f'{prefix}#[targets."{target_label.translate(escape_table)}".class_weights]\n'
        )
        for cat, weights in sorted(pos_weights.items()):
            outtoml.write(
                f'{prefix}#"{cat.translate(escape_table)}" = {weights:1.4g}\n'
            )
        outtoml.write("\n")

    # Qunatization bins for continuous variables
    for target_label, bincount in args.quantize:
        vals = pd.to_numeric(clini_df[target_label]).dropna()

        # Calculate quantization thresholds for n equally sized bins
        # -inf, +inf have to be set to finite values for np.quantile to work
        vals_clamped = vals.replace(
            {
                -np.inf: vals[vals != -np.inf].min(),
                np.inf: vals[vals != np.inf].max(),
            }
        ).dropna()
        thresholds = np.array(
            [
                -np.inf,
                *np.quantile(vals_clamped, q=np.linspace(0, 1, bincount + 1))[1:-1],
                np.inf,
            ]
        )

        _, one_hot, weights = encode_quantize(
            clini_df=clini_df, target_label=target_label, thresholds=thresholds
        )

        # We do a two-step approach:
        # First, we naively calculate the quantile thresholds (as done in the
        # above lines).  If some values appear often (more than n_samples/n_bins
        # times), some of the thresholds may be the same.  In that case, we drop
        # the superfluous classes (below).
        well_supported = one_hot.sum(0) > args.category_min_count
        thresholds = [thresholds[0], *thresholds[1:][well_supported]]
        categories, one_hot, weights = encode_quantize(
            clini_df=clini_df, target_label=target_label, thresholds=thresholds
        )

        # Badly supported target: comment out
        prefix = "#" if len(categories) < 2 else ""

        outtoml.write(f'{prefix}[targets."{target_label.translate(escape_table)}"]\n')
        outtoml.write(f'{prefix}type = "quantize"\n')

        draw_histogram(vals=vals, outtoml=outtoml, prefix=prefix)

        if not well_supported.all():
            outtoml.write(
                f"{prefix}# Too imbalanced a dataset to quantize into {bincount} bins"
            )

        # Print out a list of thresholds
        # The items outside of the lowest / highest threshold are to be
        # interpreted as NaNs
        outtoml.write(
            f"{prefix}thresholds = [ {', '.join(f'{t:+1.2e}' for t in thresholds)} ]\n"
        )

        # Write the weights
        outtoml.write(
            f'{prefix}#[targets."{target_label.translate(escape_table)}".class_weights]\n'
        )
        for category, weight, count in zip(
            categories, weights, one_hot.sum(0), strict=True
        ):
            outtoml.write(
                f'{prefix}#"{category}" = {weight:1.4g}\t# count = {int(count)}\n'
            )
        outtoml.write("\n")


def draw_histogram(*, vals: npt.NDArray[np.float_], outtoml, prefix: str):
    # vals w/o infinite values / nans
    vals_finite = vals.replace(
        {
            -np.inf: np.nan,
            np.inf: np.nan,
        }
    ).dropna()
    counts, bins = np.histogram(vals_finite)

    # Draw a histogram of all classes to give the user a feeling
    # for the distribution of the data
    outtoml.write(f"{prefix}# bin       count\n")
    # Bin exclusively for -inf values
    if (vals == -np.inf).any():
        inf_count = (vals == -np.inf).sum()
        outtoml.write(
            f"{prefix}# -inf      {inf_count:>5d} {'*'*np.round(60*inf_count / counts.max())}\n"
        )
    # Bins for finite values
    for bin, count, width in zip(
        bins, counts, np.round(60 * counts / counts.max()).astype(int)
    ):
        outtoml.write(f"{prefix}# {bin:+1.2e} {count:>5d} {'*'*width}\n")
    # Bin exclusively for inf values
    if (vals == np.inf).any():
        inf_count = (vals == np.inf).sum()
        outtoml.write(
            f"{prefix}# +inf      {inf_count:>5d} {'*'*np.round(60*inf_count / counts.max()).astype(int)}\n"
        )


if __name__ == "__main__":
    main()
