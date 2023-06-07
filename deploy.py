import argparse
import math
import re
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.loggers import CSVLogger

from train import BagDataset, LitBarspoonTransformer, read_table

# %%
# %%

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clini-table", type=Path, required=True)
    parser.add_argument("--slide-table", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--model-ckpt", type=Path, required=True)
    args = parser.parse_args()

    model = LitBarspoonTransformer.load_from_checkpoint(args.model_ckpt)

    clini_df = read_table(
        args.clini_table,
        dtype={"PATIENT": str},
    )
    slide_df = read_table(
        args.slide_table,
    )[["PATIENT", "FILENAME"]]
    df = clini_df.merge(slide_df, on="PATIENT")
    assert not df.empty, "no overlap between clini and slide table."

    # remove slides we don't have
    h5s = set(args.feature_dir.glob("*.h5"))
    assert h5s, f"no features found in {args.feature_dir}!"
    h5_df = pd.DataFrame(h5s, columns=["slide_path"])
    h5_df["FILENAME"] = h5_df.slide_path.map(lambda p: p.stem)
    cohort_df = df.merge(h5_df, on="FILENAME").reset_index()
    # reduce to one row per patient with list of slides in `df['slide_path']`
    patient_df = cohort_df.groupby("PATIENT").first().drop(columns="slide_path")
    patient_slides = cohort_df.groupby("PATIENT").slide_path.apply(list)
    cohort_df = patient_df.merge(
        patient_slides, left_on="PATIENT", right_index=True
    ).reset_index()

    targets = torch.Tensor(cohort_df[model.target_labels].apply(pd.to_numeric).values)
    bags = cohort_df.slide_path.values

    test_ds = BagDataset(
        bags,
        targets,
        instances_per_bag=2**12,
        deterministic=True,
    )
    test_dl = DataLoader(test_ds, batch_size=1, num_workers=8)

    trainer = pl.Trainer(
        accelerator="auto",
        logger=CSVLogger(save_dir=args.output_dir),
    )

    trainer.test(model=model, dataloaders=test_dl)
