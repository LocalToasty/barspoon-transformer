#!/usr/bin/env python3
import argparse
import os
import tomli
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from barspoon.data import BagDataset
from barspoon.model import LitEncDecTransformer
from barspoon.target_file import encode_targets
from barspoon.utils import flatten_batched_dicts, make_dataset_df, make_preds_df


def main():
    parser = make_argument_parser()
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    with open(args.target_file, "rb") as target_toml_file:
        target_info = tomli.load(target_toml_file)
    target_labels = list(target_info["targets"].keys())

    if args.valid_clini_tables or args.valid_slide_tables or args.valid_feature_dirs:
        # read validation set from separate clini / slide table / feature dir
        train_df = make_dataset_df(
            clini_tables=args.clini_tables,
            slide_tables=args.slide_tables,
            feature_dirs=args.feature_dirs,
            patient_col=args.patient_col,
            filename_col=args.filename_col,
            group_by=args.group_by,
            target_labels=target_labels,
        )
        valid_df = make_dataset_df(
            clini_tables=args.valid_clini_tables or args.clini_tables,
            slide_tables=args.valid_slide_tables or args.slide_tables,
            feature_dirs=args.valid_feature_dirs or args.feature_dirs,
            patient_col=args.patient_col,
            filename_col=args.filename_col,
            group_by=args.group_by,
            target_labels=target_labels,
        )
    else:
        # split validation set off main dataset
        dataset_df = make_dataset_df(
            clini_tables=args.clini_tables,
            slide_tables=args.slide_tables,
            feature_dirs=args.feature_dirs,
            patient_col=args.patient_col,
            filename_col=args.filename_col,
            group_by=args.group_by,
            target_labels=target_labels,
        )
        train_items, valid_items = train_test_split(dataset_df.index, test_size=0.2)
        train_df, valid_df = dataset_df.loc[train_items], dataset_df.loc[valid_items]

    train_encoded_targets = encode_targets(
        train_df, target_labels=target_labels, **target_info
    )

    valid_encoded_targets = encode_targets(
        valid_df, target_labels=target_labels, **target_info
    )

    assert not (
        overlap := set(train_df.index) & set(valid_df.index)
    ), f"unexpected overlap between training and testing set: {overlap}"

    train_dl, valid_dl = make_dataloaders(
        train_bags=train_df.path.values,
        train_targets={k: v.encoded for k, v in train_encoded_targets.items()},
        valid_bags=valid_df.path.values,
        valid_targets={k: v.encoded for k, v in valid_encoded_targets.items()},
        instances_per_bag=args.instances_per_bag,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    example_bags, _, _ = next(iter(train_dl))
    d_features = example_bags.size(-1)

    model = LitEncDecTransformer(
        d_features=d_features,
        target_labels=target_labels,
        weights={k: v.weight for k, v in train_encoded_targets.items()},
        # Other hparams
        version="barspoon-transformer 3.0",
        categories={k: v.categories for k, v in train_encoded_targets.items()},
        target_file=target_info,
        **{
            f"train_{train_df.index.name}": list(train_df.index),
            f"valid_{valid_df.index.name}": list(valid_df.index),
        },
        **{k: v for k, v in vars(args).items() if k not in {"target_file"}},
    )

    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=args.patience),
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                filename="checkpoint-{epoch:02d}-{val_loss:0.3f}",
            ),
        ],
        max_epochs=args.max_epochs,
        # FIXME The number of accelerators is currently fixed to one for the
        # following reasons:
        #  1. `trainer.predict()` does not return any predictions if used with
        #     the default strategy no multiple GPUs
        #  2. `barspoon.model.SafeMulticlassAUROC` breaks on multiple GPUs.
        accelerator=args.accelerator,
        devices=1,
        accumulate_grad_batches=args.accumulate_grad_samples // args.batch_size,
        gradient_clip_val=0.5,
        logger=CSVLogger(save_dir=args.output_dir),
    )

    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=valid_dl)

    predictions = flatten_batched_dicts(
        trainer.predict(model=model, dataloaders=valid_dl, return_predictions=True)
    )

    preds_df = make_preds_df(
        predictions=predictions,
        base_df=valid_df,
        categories={k: v.categories for k, v in train_encoded_targets.items()},
    )
    preds_df.to_csv(args.output_dir / "valid-patient-preds.csv")


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output-dir",
        metavar="OUTPUT_DIR",
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
        required=True,
        action="append",
        help="Path to the clinical table. Can be specified multiple times",
    )
    parser.add_argument(
        "-s",
        "--slide-table",
        metavar="SLIDE_TABLE",
        dest="slide_tables",
        type=Path,
        required=True,
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

    valid_cohort_parser = parser.add_argument_group(title="optional validation cohort")
    valid_cohort_parser.add_argument(
        "--valid-clini-table",
        metavar="PATH",
        dest="valid_clini_tables",
        type=Path,
        action="append",
        help="Path to the clinical table of the validation set. Can be specified multiple times",
    )
    valid_cohort_parser.add_argument(
        "--valid-slide-table",
        metavar="PATH",
        dest="valid_slide_tables",
        type=Path,
        action="append",
        help="Path to the slide table of the validation set. Can be specified multiple times",
    )
    valid_cohort_parser.add_argument(
        "--valid-feature-dir",
        metavar="PATH",
        dest="valid_feature_dirs",
        type=Path,
        action="append",
        help="Path containing the slide features of the validation set as `h5` files. Can be specified multiple times",
    )

    parser.add_argument(
        "--target-file",
        metavar="PATH",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--patient-col",
        metavar="COL",
        type=str,
        default="patient",
        help="Name of the patient column",
    )
    parser.add_argument(
        "--filename-col",
        metavar="COL",
        type=str,
        default="filename",
        help="Name of the slide column",
    )
    parser.add_argument(
        "--group-by",
        metavar="COL",
        type=str,
        help="How to group slides. If 'clini' table is given, default is 'patient'; otherwise, default is 'slide'",
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
    training_parser.add_argument("--seed", type=int, default=0)
    training_parser.add_argument("--accelerator", type=str, default="auto")

    return parser


def make_dataloaders(
    *,
    train_bags: Sequence[Iterable[Path]],
    train_targets: Mapping[str, torch.Tensor],
    valid_bags: Sequence[Iterable[Path]],
    valid_targets: Mapping[str, torch.Tensor],
    batch_size: int,
    instances_per_bag: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = BagDataset(
        bags=train_bags,
        targets=train_targets,
        instances_per_bag=instances_per_bag,
        deterministic=False,
    )
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )

    valid_ds = BagDataset(
        bags=valid_bags,
        targets=valid_targets,
        instances_per_bag=instances_per_bag,
        deterministic=True,
    )
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, num_workers=num_workers)

    return train_dl, valid_dl


if __name__ == "__main__":
    main()
