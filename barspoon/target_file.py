"""Automatically generate target information from clini table"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from packaging.specifiers import Specifier

from barspoon.utils import read_table

__all__ = ["encode_targets", "decode_targets"]


class EncodedTarget(NamedTuple):
    categories: List[str]
    encoded: torch.Tensor
    weight: torch.Tensor


def encode_targets(
    clini_df: pd.DataFrame,
    *,
    target_labels: Iterable[str],
    # From targets toml
    version: str = "barspoon-targets 1.0",
    targets: Dict[str, Any],
    **ignored,
) -> Dict[str, EncodedTarget]:
    """Encodes the information in a clini table into a tensor

    Returns:
        A tuple consisting of
         1. The encoded targets
         2. The categories' representatives #TODO elaborate
         3. A list of the targets' classes' weights
    """
    # Make sure target file has the right version
    name, version = version.split(" ")
    if not (
        name == "barspoon-targets" and (spec := Specifier("~=1.0")).contains(version)
    ):
        raise ValueError(
            f"incompatible target file: expected barspoon-targets{spec}, found `{name} {version}`"
        )

    if ignored:
        logging.warn(f"ignored {ignored}")

    encoded_targets = {}
    for target_label in target_labels:
        info = targets[target_label]

        if "categories" in info:
            representatives, encoded, weight = encode_category(
                clini_df=clini_df, target_label=target_label, **info
            )
            encoded_targets[target_label] = EncodedTarget(
                categories=representatives, encoded=encoded, weight=weight
            )

        elif "thresholds" in info:
            representatives, encoded, weight = encode_quantize(
                clini_df=clini_df, target_label=target_label, **info
            )
            encoded_targets[target_label] = EncodedTarget(
                categories=representatives, encoded=encoded, weight=weight
            )

        else:
            logging.warn(f"ignoring unrecognized target type {target_label}")

    return encoded_targets


def encode_category(
    *,
    clini_df: pd.DataFrame,
    target_label: str,
    categories: Sequence[List[str]],
    class_weights: Optional[Dict[str, float]] = None,
    **ignored,
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
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
    thresholds: List[float],
    class_weights: Optional[Dict[str, float]] = None,
    **ignored,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    # Warn user of unused labels
    if ignored:
        logging.warn(f"ignored labels in target {target_label}: {ignored}")

    n_bins = len(thresholds) - 1
    numeric_vals = torch.tensor(pd.to_numeric(clini_df[target_label]).values).view(
        -1, 1
    )
    # Count the number of left bounds of bins each item is greater or equal than
    # This will lead to nans being mapped to 0,
    # whereas each other classes will be set to C+1
    bin_index = (numeric_vals >= torch.tensor(thresholds).view(1, -1)).count_nonzero(
        dim=1
    )
    # One hot encode and discard nan columns (first and last col)
    # This also maps all classes from C+1 back to C
    one_hot = F.one_hot(bin_index, num_classes=n_bins + 2)[:, 1:-1]

    # Class weights
    categories = [
        f"[{l:+1.2e};{u:+1.2e})" for l, u in zip(thresholds[:-1], thresholds[1:])
    ]
    if class_weights is not None:
        weight = torch.tensor([class_weights[c] for c in categories])
    else:
        # No class weights given; use 1/#bins
        weight = torch.tensor([1 / n_bins] * n_bins)

    return categories, one_hot, weight


def decode_targets(
    encoded: torch.Tensor,
    *,
    target_labels: Sequence[str],
    targets: Dict[str, Any],
    version: str = "barspoon-targets 1.0",
    **ignored,
) -> List[np.array]:
    name, version = version.split(" ")
    if not (
        name == "barspoon-targets" and (spec := Specifier("~=1.0")).contains(version)
    ):
        raise ValueError(f"model not compatible with barspoon-targets {spec}", version)

    # Warn user of unused labels
    if ignored:
        logging.warn(f"ignored parameters: {ignored}")

    decoded_targets = []
    curr_col = 0
    for target_label in target_labels:
        info = targets[target_label]

        if (categories := info.get("categories")) is not None:
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

        elif (thresholds := info.get("thresholds")) is not None:
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
            raise ValueError(f"cannot decode {target_label}: no target info")

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

    # Artifact name and version (follows semantic versioning)
    outtoml.write('version = "barspoon-targets 1.0"\n\n')

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
        if prefix:
            outtoml.write(
                "## WARNING: fewer than two classes with sufficient #samples\n"
            )

        # List all categories
        # with little-populated categories being commented out
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
        prefix = "#"

        well_supported_counts = counts[counts >= args.category_min_count]
        pos_weights = well_supported_counts.sum() / well_supported_counts
        pos_weights /= pos_weights.sum()
        outtoml.write(
            f'{prefix}[targets."{target_label.translate(escape_table)}".class_weights]\n'
        )
        for cat, weight in sorted(pos_weights.items()):
            outtoml.write(f'{prefix}"{cat.translate(escape_table)}" = {weight:1.4g}\n')
        outtoml.write("\n")

    # Qunatization bins for continuous variables
    for target_label, bincount in args.quantize:
        vals = pd.to_numeric(clini_df[target_label]).dropna()
        # vals w/o infinite values / nans
        vals_finite = vals.replace(
            {
                -np.inf: np.nan,
                np.inf: np.nan,
            }
        ).dropna()
        counts, bins = np.histogram(vals_finite)

        # Comment out this section if the bins are too small
        prefix = "#" if len(vals_finite) // bincount < args.category_min_count else ""

        # Draw a histogram of all classes to give the user a feeling
        # for the distribution of the data
        outtoml.write(f'{prefix}[targets."{target_label.translate(escape_table)}"]\n')
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

        # Calculate quantization thresholds for n equally sized bins
        # -inf, +inf have to be set to finite values for np.quantile to work
        vals_clamped = vals.replace(
            {
                -np.inf: vals[vals != -np.inf].min(),
                np.inf: vals[vals != np.inf].max(),
            }
        ).dropna()
        thresholds = [
            -np.inf,
            *np.quantile(vals_clamped, q=np.linspace(0, 1, bincount + 1))[1:-1],
            np.inf,
        ]

        # Just a list of thresholds
        # The items outside of the lowest / highest threshold are to be
        # interpreted as NaNs
        outtoml.write(
            f"{prefix}thresholds = [ {', '.join(f'{t:+1.2e}' for t in thresholds)} ]\n"
        )

        # Class weights are just 1/N where N is the number of bins
        prefix = "#"
        outtoml.write(
            f'{prefix}[targets."{target_label.translate(escape_table)}".class_weights]\n'
        )
        for lower, upper in zip(thresholds[:-1], thresholds[1:]):
            outtoml.write(
                f'{prefix}"[{lower:+1.2e};{upper:+1.2e})" = {1/bincount:1.4g}\n'
            )
        outtoml.write("\n")


if __name__ == "__main__":
    main()
