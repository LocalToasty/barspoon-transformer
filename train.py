#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clini-table", type=Path, required=True)
    parser.add_argument("--slide-table", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument(
        "--target-label", type=str, required=True, action="append", dest="target_labels"
    )
    model_parser = parser.add_argument_group("model options")
    model_parser.add_argument("--num-encoder-heads", type=int, default=8)
    model_parser.add_argument("--num-decoder-heads", type=int, default=8)
    model_parser.add_argument("--num-encoder-layers", type=int, default=2)
    model_parser.add_argument("--num-decoder-layers", type=int, default=2)
    model_parser.add_argument("--d-model", type=int, default=512)
    model_parser.add_argument("--dim-feedforward", type=int, default=2048)
    training_parser = parser.add_argument_group("training options")
    training_parser.add_argument("--instances-per-bag", type=int, default=2**12)
    training_parser.add_argument("--learning-rate", type=float, default=1e-4)
    training_parser.add_argument("--batch-size", type=int, default=4)
    training_parser.add_argument("--accumulate-grad-samples", type=int, default=32)
    training_parser.add_argument(
        "--num-workers", type=int, default=min(os.cpu_count() or 0, 8)
    )
    training_parser.add_argument("--patience", type=int, default=16)
    training_parser.add_argument("--max-epochs", type=int, default=256)
    args = parser.parse_args()

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data import BagDataset
from model import LitEncDecTransformer


def read_table(path: Path, dtype=str) -> pd.DataFrame:
    if Path(path).suffix == ".csv":
        return pd.read_csv(path, dtype=dtype)
    else:
        return pd.read_excel(path, dtype=dtype)


if __name__ == "__main__":
    pl.seed_everything(0)
    torch.set_float32_matmul_precision("medium")

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

    assert len(cohort_df["PATIENT"]) == cohort_df["PATIENT"].nunique()

    # TODO fail deadly
    target_labels = np.array(args.target_labels)
    target_labels = target_labels[cohort_df[target_labels].nunique(dropna=True) == 2]
    target_labels = (
        cohort_df[target_labels]
        .select_dtypes(["int16", "int32", "int64", "float16", "float32", "float64"])
        .columns.values
    )

    targets = torch.tensor(
        cohort_df[target_labels].apply(pd.to_numeric).values, dtype=torch.float32
    )
    bags = cohort_df.slide_path.values
    pos_samples = targets.nansum(dim=0)
    neg_samples = (1 - targets).nansum(dim=0)
    pos_weight = neg_samples / pos_samples

    train_idx, valid_idx = train_test_split(np.arange(len(targets)), test_size=0.2)

    if (args.output_dir / "done").exists():
        # already done...
        exit(0)
    elif args.output_dir.exists():
        # previous attempt didn't finish; start over
        shutil.rmtree(args.output_dir)

    train_ds = BagDataset(
        bags[train_idx],
        targets[train_idx],
        instances_per_bag=args.instances_per_bag,
        deterministic=False,
    )
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
    )

    valid_ds = BagDataset(
        bags[valid_idx],
        targets[valid_idx],
        instances_per_bag=args.instances_per_bag,
        deterministic=True,
    )
    valid_dl = DataLoader(
        valid_ds, batch_size=args.batch_size, num_workers=args.num_workers
    )

    example_bags, _ = next(iter(train_dl))
    d_features = example_bags.size(-1)

    model = LitEncDecTransformer(
        d_features=d_features,
        n_targets=len(target_labels),
        pos_weight=pos_weight,
        # other hparams
        train_patients=list(cohort_df.loc[train_idx].PATIENT),
        valid_patients=list(cohort_df.loc[valid_idx].PATIENT),
        **vars(args),
    )

    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        callbacks=[
            EarlyStopping(
                monitor="val_TopKMultilabelAUROC", mode="max", patience=args.patience
            ),
            ModelCheckpoint(
                monitor="val_TopKMultilabelAUROC",
                mode="max",
                filename="checkpoint-{epoch:02d}-{val_TopKMultilabelAUROC:0.3f}",
            ),
        ],
        max_epochs=args.max_epochs,
        accelerator="auto",
        accumulate_grad_batches=args.accumulate_grad_samples // args.batch_size,
        gradient_clip_val=0.5,
        logger=CSVLogger(save_dir=args.output_dir),
    )

    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=valid_dl)

    predictions = torch.cat(trainer.predict(model=model, dataloaders=valid_dl))  # type: ignore
    preds_df = cohort_df.iloc[valid_idx][["PATIENT", *target_labels]]
    for target_label, score in zip(target_labels, predictions.transpose(1, 0)):
        preds_df[f"{target_label}_1"] = score
        preds_df[f"{target_label}_0"] = 1 - score

    preds_df.to_csv(args.output_dir / "valid-patient-preds.csv", index=False)

    with open(args.output_dir / "done", "w"):
        pass
