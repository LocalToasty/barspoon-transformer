from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd


def generate_dataset_df(
    *,
    clini_table: Optional[Path] = None,
    slide_table: Optional[Path] = None,
    feature_dir: Path,
    patient_col: str,
    slide_col: str,
    group_by: Optional[str] = None,
    target_labels: Sequence[str],
) -> pd.DataFrame:
    # Create a table mapping slide names to their paths
    h5s = list(feature_dir.glob("*.h5"))
    assert h5s, f"no features found in {feature_dir}!"
    h5_df = pd.DataFrame(h5s, columns=["path"])
    h5_df["slide"] = h5_df.path.map(lambda p: p.stem)
    h5_df = h5_df.set_index("slide", verify_integrity=True)
    df = h5_df

    if slide_table is not None:
        slide_df = read_table(
            slide_table,
        )[
            [patient_col, slide_col]
        ].rename(columns={patient_col: "patient", slide_col: "slide"})
        slide_df = slide_df.set_index("slide", verify_integrity=True)
        df = df.join(slide_df, how="inner").reset_index()

    if clini_table is not None:
        assert (
            slide_table is not None
        ), "--slide-table has to be specified to associate clinical information with slides"

        clini_df = (
            read_table(
                clini_table,
                dtype={patient_col: str},
            )
            .rename(columns={patient_col: "patient"})
            .set_index("patient", verify_integrity=True)
        )
        # select all the relevant available ground truths
        clini_df = clini_df[list(set(target_labels) & set(clini_df.columns))]
        df = df.merge(clini_df.reset_index(), on="patient")

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
        return pd.read_csv(table, dtype=dtype, low_memory=False)
    else:
        return pd.read_excel(table, dtype=dtype, low_memory=False)
