from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import pandas as pd
import torch


def make_dataset_df(
    *,
    clini_tables: Iterable[Path] = [],
    slide_tables: Iterable[Path] = [],
    feature_dirs: Iterable[Path],
    patient_col: Optional[str] = None,
    slide_col: Optional[str] = None,
    group_by: Optional[str] = None,
    target_labels: Sequence[str],
) -> pd.DataFrame:
    # Create a table mapping slide names to their paths
    h5s = {h5 for d in feature_dirs for h5 in d.glob("*.h5")}
    assert h5s, f"no features found in {feature_dirs}!"
    h5_df = pd.DataFrame(list(h5s), columns=["path"])
    h5_df["slide"] = h5_df.path.map(lambda p: p.stem)
    h5_df = h5_df.set_index("slide", verify_integrity=True)
    df = h5_df

    if slide_tables:
        slide_df = pd.concat([read_table(slide_table) for slide_table in slide_tables])
        patient_col = patient_col or "PATIENT" if "PATIENT" in slide_df else "patient"
        slide_col = slide_col or "FILENAME" if "FILENAME" in slide_df else "slide"
        slide_df = slide_df[[patient_col, slide_col]].rename(
            columns={patient_col: "patient", slide_col: "slide"}
        )
        slide_df = slide_df.drop_duplicates().set_index("slide", verify_integrity=True)
        df = df.join(slide_df, how="inner").reset_index()
        assert not df.empty, "no match between features and slide table"

    if clini_tables:
        assert (
            slide_tables
        ), "--slide-table has to be specified to associate clinical information with slides"

        clini_df = pd.concat(
            [
                read_table(
                    clini_table,
                    dtype={patient_col: str},
                )
                for clini_table in clini_tables
            ]
        ).rename(columns={patient_col: "patient"})
        # select all the relevant available ground truths,
        # make sure there's no conflicting patient info
        clini_df = (
            clini_df[["patient", *list(set(target_labels) & set(clini_df.columns))]]
            .drop_duplicates()
            .set_index("patient", verify_integrity=True)
        )
        df = df.merge(clini_df.reset_index(), on="patient")
        assert not df.empty, "no match between slides and clini table"

    # At this point we have a dataframe containing
    # - h5 paths
    # - the corresponding slide names
    # - the patient id (if a slide table was given)
    # - the ground truths for the target labels present in the clini table

    group_by = group_by or "patient" if "patient" in df else "slide"

    # Group paths and metadata by the specified column
    grouped_paths_df = df.groupby(group_by)[["path"]].aggregate(list)
    grouped_metadata_df = (
        df.groupby(group_by).first().drop(columns=["path", "slide"], errors="ignore")
    )
    df = grouped_metadata_df.join(grouped_paths_df)

    return df


def read_table(table: Union[Path, pd.DataFrame], dtype=str) -> pd.DataFrame:
    if isinstance(table, pd.DataFrame):
        return table

    if table.suffix == ".csv":
        return pd.read_csv(table, dtype=dtype, low_memory=False)  # type: ignore
    else:
        return pd.read_excel(table, dtype=dtype, low_memory=False)  # type: ignore


def make_preds_df(
    predictions: torch.Tensor,
    weight: Optional[torch.Tensor],
    pos_weight: Optional[torch.Tensor],
    base_df: pd.DataFrame,
    target_labels: Sequence[str],
) -> pd.DataFrame:
    preds_df = pd.concat(
        [
            base_df[target_labels],
            *[
                pd.DataFrame(
                    {f"{target_label}_1": score, f"{target_label}_0": 1 - score},
                    index=base_df.index,
                )
                for target_label, score in zip(
                    target_labels, predictions.transpose(1, 0)
                )
            ],
        ],
        axis=1,
    ).copy()

    # all target labels for which we have clinical information
    has_ground_truth = [t in base_df.columns for t in target_labels]

    ys = predictions[:, has_ground_truth]
    ts = torch.tensor(preds_df[target_labels].values)
    # calculate the element-wise loss
    preds_df["loss"] = torch.nn.functional.binary_cross_entropy_with_logits(
        input=ys.where(~ts.isnan(), 0),
        target=ts.where(~ts.isnan(), 0),
        weight=weight[has_ground_truth] if weight is not None else None,
        pos_weight=pos_weight[has_ground_truth] if pos_weight is not None else None,
        reduction="none",
    ).nanmean(dim=1)

    preds_df = preds_df.sort_values(by="loss")

    return preds_df
