"""Automatically generate target information from clini table"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from barspoon.utils import read_table


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
    outtoml.write('version = "barspoon-targets 1.0-pre1"\n\n')

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
