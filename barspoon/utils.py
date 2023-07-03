import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from packaging.version import Version

__all__ = [
    "make_dataset_df",
    "read_table",
    "make_preds_df",
    "note_problem",
    "encode_targets",
    "decode_targets",
]


def make_dataset_df(
    *,
    clini_tables: Iterable[Path] = [],
    slide_tables: Iterable[Path] = [],
    feature_dirs: Iterable[Path],
    patient_col: str = "patient",
    filename_col: str = "filename",
    group_by: Optional[str] = None,
    target_labels: Sequence[str],
) -> pd.DataFrame:
    if slide_tables:
        slide_dfs = []
        for slide_table in slide_tables:
            slide_df = read_table(slide_table)
            slide_df = slide_df.loc[
                :, slide_df.columns.isin([patient_col, filename_col])  # type: ignore
            ]

            assert filename_col in slide_df, (
                f"{filename_col} not in {slide_table}. "
                "Use `--filename-col <COL>` to specify a different column name"
            )
            slide_df["path"] = slide_df[filename_col].map(
                lambda fn: next(
                    (path for f in feature_dirs if (path := f / fn).exists()), None
                )
            )

            if (na_idxs := slide_df.path.isna()).any():
                note_problem(
                    f"some slides from {slide_table} have no features: {list(slide_df.loc[na_idxs, filename_col])}",
                    "warn",
                )
            slide_df = slide_df[~na_idxs]
            slide_dfs.append(slide_df)
            assert not slide_df.empty, f"no features for slide table {slide_table}"

        df = pd.concat(slide_dfs)
    else:
        # Create a table mapping slide names to their paths
        h5s = {h5 for d in feature_dirs for h5 in d.glob("*.h5")}
        assert h5s, f"no features found in {feature_dirs}!"
        df = pd.DataFrame(list(h5s), columns=["path"])
        df[filename_col] = df.path.map(lambda p: p.name)

    # df is now a DataFrame containing at least a column "path", possibly a patient and filename column

    if clini_tables:
        assert patient_col in df.columns, (
            f"a slide table with {patient_col} column has to be specified using `--slide-table <PATH>` "
            "or the patient column has to be specified with `--patient-col <COL>`"
        )

        clini_df = pd.concat([read_table(clini_table) for clini_table in clini_tables])
        # select all the relevant available ground truths,
        # make sure there's no conflicting patient info
        clini_df = (
            # select all important columns
            clini_df.loc[
                :, clini_df.columns.isin([patient_col, group_by, *target_labels])  # type: ignore
            ]
            .drop_duplicates()
            .set_index(patient_col, verify_integrity=True)
        )
        # TODO assert patient_col in clini_df, f"no column named {patient_col} in {clini_df}"
        df = df.merge(clini_df.reset_index(), on=patient_col)
        assert not df.empty, "no match between slides and clini table"

    # At this point we have a dataframe containing
    # - h5 paths
    # - the corresponding slide names
    # - the patient id (if a slide table was given)
    # - the ground truths for the target labels present in the clini table

    group_by = group_by or patient_col if patient_col in df else filename_col

    # Group paths and metadata by the specified column
    grouped_paths_df = df.groupby(group_by)[["path"]].aggregate(list)
    grouped_metadata_df = (
        df.groupby(group_by)
        .first()
        .drop(columns=["path", filename_col], errors="ignore")
    )
    df = grouped_metadata_df.join(grouped_paths_df)

    return df


def read_table(table: Union[Path, pd.DataFrame], dtype=str) -> pd.DataFrame:
    if isinstance(table, pd.DataFrame):
        return table

    if table.suffix == ".csv":
        return pd.read_csv(table, dtype=dtype, low_memory=False)  # type: ignore
    else:
        return pd.read_excel(table, dtype=dtype)  # type: ignore


def make_preds_df(
    predictions: torch.Tensor,
    *,
    base_df: pd.DataFrame,
    target_labels: Sequence[str],
    categories: Sequence[str],
) -> pd.DataFrame:
    assert len(target_labels) == len(categories)

    target_edges = np.cumsum([0, *(len(c) for c in categories)])

    target_pred_dfs = []
    for target_label, cat_labels, left, right in zip(
        target_labels, categories, target_edges[:-1], target_edges[1:]
    ):
        target_pred_df = pd.DataFrame(
            predictions[:, left:right],
            columns=[f"{target_label}_{cat}" for cat in cat_labels],
            index=base_df.index,
        )
        hard_prediction = np.array(cat_labels)[predictions[:, left:right].argmax(dim=1)]
        target_pred_df[f"{target_label}_pred"] = hard_prediction

        target_pred_dfs.append(target_pred_df)

    preds_df = pd.concat(
        [base_df.loc[:, base_df.columns.isin(target_labels)], *target_pred_dfs], axis=1
    ).copy()
    return preds_df


def note_problem(msg, mode: Literal["raise", "warn", "ignore"]):
    if mode == "raise":
        raise RuntimeError(msg)
    elif mode == "warn":
        logging.warning(msg)
    elif mode == "ignore":
        return
    else:
        raise ValueError("unknown error propagation type", mode)


def encode_targets(
    clini_df: pd.DataFrame,
    *,
    target_labels: Iterable[str],
    # From targets toml
    version: str = "barspoon-targets 1.0",
    targets: Dict[str, Any],
    **ignored,
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    """Encodes the information in a clini table into a tensor

    Returns:
        A tuple consisting of
         1. The encoded targets
         2. The categories' representative  #TODO elaborate
         3. A list of the targets' classes' weights
    """
    # Make sure target file has the right version
    name, version = version.split(" ")
    assert name == "barspoon-targets"
    version = Version(version)
    # assert version >= Version("1.0") and version < Version("2.0")
    assert version == Version("1.0-pre1")

    if ignored:
        logging.warn(f"ignored {ignored}")

    all_representatives, encoded_cols, weights = [], [], []
    for target_label in target_labels:
        info = targets[target_label]

        if "categories" in info:
            representatives, encoded, weight = encode_category(
                clini_df=clini_df, target_label=target_label, **info
            )
            all_representatives.append(representatives)
            encoded_cols.append(encoded)
            weights.append(weight)

        elif "thresholds" in info:
            representatives, encoded, weight = encode_quantize(
                clini_df=clini_df, target_label=target_label, **info
            )
            all_representatives.append(representatives)
            encoded_cols.append(encoded)
            weights.append(weight)

        else:
            logging.warn(f"ignoring unrecognized target type {target_label}")

    assert len(encoded_cols) == len(weights)
    return torch.cat(encoded_cols, dim=1), all_representatives, weights


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
    assert name == "barspoon-targets"
    version = Version(version)
    # assert version >= Version("1.0") and version < Version("2.0")
    assert version == Version("1.0-pre1")

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
