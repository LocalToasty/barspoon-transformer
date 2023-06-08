#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .data import BagDataset
from .model import LitEncDecTransformer
from .utils import generate_dataset_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="Directory path for the output",
    )
    parser.add_argument(
        "-c",
        "--clini-table",
        metavar="PATH",
        type=Path,
        help="Path to the clinical table",
    )
    parser.add_argument(
        "-s", "--slide-table", metavar="PATH", type=Path, help="Path to the slide table"
    )
    parser.add_argument(
        "-f",
        "--feature-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="Path containing the slide features as `h5` files",
    )
    parser.add_argument(
        "-m",
        "--checkpoint-path",
        metavar="PATH",
        type=Path,
        required=True,
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--patient-col",
        metavar="COL",
        type=str,
        default="PATIENT",
        help="Name of the patient column",
    )
    parser.add_argument(
        "--slide-col",
        metavar="COL",
        type=str,
        default="FILENAME",
        help="Name of the slide column",
    )
    parser.add_argument(
        "--group-by",
        metavar="COL",
        type=str,
        help="How to group slides. If 'clini' table is given, default is 'patient'; otherwise, default is 'slide'",
    )
    parser.add_argument("--num-workers", type=int, default=min(os.cpu_count() or 0, 8))
    args = parser.parse_args()

    pl.seed_everything(0)
    torch.set_float32_matmul_precision("medium")

    model = LitEncDecTransformer.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path
    )
    target_labels = model.hparams["target_labels"]

    df = generate_dataset_df(
        clini_table=args.clini_table,
        slide_table=args.slide_table,
        feature_dir=args.feature_dir,
        patient_col=args.patient_col,
        slide_col=args.slide_col,
        group_by=args.group_by,
        target_labels=target_labels,
    )

    # make a dataset with faux labels (the labels will be ignored)
    ds = BagDataset(
        bags=list(df.path), targets=torch.zeros(len(df), 0), instances_per_bag=None
    )
    dl = DataLoader(ds, shuffle=False, num_workers=args.num_workers)

    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        accelerator="auto",
        devices=1,
    )
    predictions = torch.cat(trainer.predict(model=model, dataloaders=dl))  # type: ignore

    preds_df = df.drop(columns="path")
    for target_label, preds in zip(target_labels, predictions.transpose(1, 0)):
        preds_df[f"{target_label}_0"] = 1 - preds
        preds_df[f"{target_label}_1"] = preds

    # calculate the element-wise loss
    weight = model.loss.weight
    pos_weight = model.loss.pos_weight

    # all target labels for which we have clinical information
    has_info = [t in df.columns for t in target_labels]

    preds_df["loss"] = torch.nn.functional.binary_cross_entropy_with_logits(
        predictions,
        predictions,
        weight=weight[has_info] if weight is not None else None,
        pos_weight=pos_weight[has_info] if pos_weight is not None else None,
        reduction="none",
    ).nanmean(dim=1)
    preds_df = preds_df.sort_values(by="loss")

    # save results
    args.output_dir.mkdir(exist_ok=True, parents=True)
    preds_df.to_csv(args.output_dir / "patient-preds.csv")


if __name__ == "__main__":
    main()
