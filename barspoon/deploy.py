#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from packaging.specifiers import SpecifierSet
from torch.utils.data import DataLoader

from barspoon.data import BagDataset
from barspoon.model import LitEncDecTransformer
from barspoon.utils import flatten_batched_dicts, make_dataset_df, make_preds_df


def main():
    parser = make_argument_parser()
    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")

    model = LitEncDecTransformer.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path
    )
    name, version = model.hparams.get("version", "undefined 0").split(" ")
    if not (
        name == "barspoon-transformer"
        and (spec := SpecifierSet(">=3.0,<4")).contains(version)
    ):
        raise ValueError(
            f"model not compatible. Found {name} {version}, expected barspoon-transformer {spec}"
        )

    target_labels = model.hparams["target_labels"]

    dataset_df = make_dataset_df(
        clini_tables=args.clini_tables,
        slide_tables=args.slide_tables,
        feature_dirs=args.feature_dirs,
        patient_col=args.patient_col,
        filename_col=args.filename_col,
        group_by=args.group_by,
        target_labels=target_labels,
    )

    # Make a dataset with faux labels (the labels will be ignored)
    ds = BagDataset(
        bags=list(dataset_df.path),
        targets={},
        instances_per_bag=None,
    )
    dl = DataLoader(ds, shuffle=False, num_workers=args.num_workers)

    # FIXME The number of accelerators is currently fixed because
    # `trainer.predict()` does not return any predictions if used with the
    # default strategy no multiple GPUs
    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        accelerator=args.accelerator,
        devices=1,
        logger=False,
    )
    predictions = flatten_batched_dicts(trainer.predict(model=model, dataloaders=dl))
    preds_df = make_preds_df(
        predictions=predictions,
        base_df=dataset_df.drop(columns="path"),
        categories=model.hparams["categories"],
    )

    # save results
    args.output_dir.mkdir(exist_ok=True, parents=True)
    preds_df.to_csv(args.output_dir / "patient-preds.csv")


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory path for the output",
    )

    parser.add_argument(
        "-c",
        "--clini-table",
        metavar="CLINI_TABLE",
        dest="clini_tables",
        type=Path,
        action="append",
        help="Path to the clinical table. Can be specified multiple times",
    )
    parser.add_argument(
        "-s",
        "--slide-table",
        dest="slide_tables",
        metavar="SLIDE_TABLE",
        type=Path,
        action="append",
        help="Path to the slide table. Can be specified multiple times",
    )
    parser.add_argument(
        "-f",
        "--feature-dir",
        metavar="FEATURE_DIR",
        dest="feature_dirs",
        type=Path,
        required=True,
        action="append",
        help="Path containing the slide features as `h5` files. Can be specified multiple times",
    )

    parser.add_argument(
        "-m",
        "--checkpoint-path",
        type=Path,
        required=True,
        help="Path to the checkpoint file",
    )

    parser.add_argument(
        "--patient-col",
        type=str,
        default="patient",
        help="Name of the patient column",
    )
    parser.add_argument(
        "--filename-col",
        type=str,
        default="filename",
        help="Name of the slide column",
    )

    parser.add_argument(
        "--group-by",
        type=str,
        help="How to group slides. If 'clini' table is given, default is 'patient'; otherwise, default is 'slide'",
    )

    parser.add_argument("--num-workers", type=int, default=min(os.cpu_count() or 0, 8))
    parser.add_argument("--accelerator", type=str, default="auto")

    return parser


if __name__ == "__main__":
    main()
