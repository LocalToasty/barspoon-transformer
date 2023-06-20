#!/usr/bin/env python3
# %%
import argparse
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from .data import BagDataset
from .model import LitEncDecTransformer
from .utils import filter_targets, get_pos_weight, make_dataset_df, make_preds_df


def main():
    parser = make_argument_parser()
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    # read target labels from file, if need be
    if args.target_labels:
        target_labels = args.target_labels
    else:
        with open(args.target_file) as f:
            target_labels = [l.strip() for l in f if l]

    dataset_df = make_dataset_df(
        clini_tables=args.clini_tables,
        slide_tables=args.slide_tables,
        feature_dirs=args.feature_dirs,
        patient_col=args.patient_col,
        filename_col=args.filename_col,
        group_by=args.group_by,
        target_labels=target_labels,
    )

    for fold_no, (train_idx, valid_idx, test_idx) in enumerate(
        get_splits(dataset_df.index.values)
    ):
        fold_dir = args.output_dir / f"{fold_no=}"
        fold_dir.mkdir(exist_ok=True, parents=True)
        train_df, valid_df, test_df = (
            dataset_df.loc[train_idx],
            dataset_df.loc[valid_idx],
            dataset_df.loc[test_idx],
        )

        # see if target labels are good, otherwise complain / die a fiery death
        # depending on whether `--filter-targets` was set by the the user
        target_labels = filter_targets(
            train_df=train_df,
            target_labels=np.array(target_labels),
            mode="warn" if args.filter_targets else "raise",
        )

        pos_weight = get_pos_weight(
            torch.tensor(
                train_df[target_labels].apply(pd.to_numeric).values, dtype=torch.float32
            )
        )

        assert not (
            overlap := set(train_df.index) & set(valid_df.index)
        ), f"overlap between training and testing set: {overlap}"

        train_dl, valid_dl, test_dl = make_dataloaders(
            train_bags=train_df.path.values,
            train_targets=torch.tensor(train_df[target_labels].apply(pd.to_numeric).values),  # type: ignore
            valid_bags=valid_df.path.values,
            valid_targets=torch.tensor(valid_df[target_labels].apply(pd.to_numeric).values),  # type: ignore
            test_bags=test_df.path.values,
            test_targets=torch.tensor(test_df[target_labels].apply(pd.to_numeric).values),  # type: ignore
            instances_per_bag=args.instances_per_bag,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        example_bags, _ = next(iter(train_dl))
        d_features = example_bags.size(-1)

        model = LitEncDecTransformer(
            d_features=d_features,
            target_labels=target_labels,
            pos_weight=pos_weight,
            # other hparams
            **{
                f"train_{train_df.index.name}": list(train_df.index),
                f"valid_{valid_df.index.name}": list(valid_df.index),
                f"test_{test_df.index.name}": list(test_df.index),
            },
            **{k: v for k, v in vars(args).items() if k not in {"target_labels"}},
        )

        trainer = pl.Trainer(
            default_root_dir=fold_dir,
            callbacks=[
                EarlyStopping(
                    monitor="val_TopKMultilabelAUROC",
                    mode="max",
                    patience=args.patience,
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
            logger=CSVLogger(save_dir=fold_dir),
        )

        trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=valid_dl)

        # save best validation set predictions
        valid_preds = torch.cat(trainer.predict(model=model, dataloaders=valid_dl, return_predictions=True))  # type: ignore
        valid_preds_df = make_preds_df(
            predictions=valid_preds,
            weight=None,  # TODO
            pos_weight=model.pos_weight,
            base_df=valid_df,
            target_labels=target_labels,
        )
        valid_preds_df.to_csv(fold_dir / "valid-patient-preds.csv")

        # save test set predictions
        test_preds = torch.cat(trainer.predict(model=model, dataloaders=test_dl, return_predictions=True))  # type: ignore
        test_preds_df = make_preds_df(
            predictions=test_preds,
            weight=None,  # TODO
            pos_weight=model.pos_weight,
            base_df=test_df,
            target_labels=target_labels,
        )
        test_preds_df.to_csv(fold_dir / "patient-preds.csv")


def make_argument_parser() -> argparse.ArgumentParser:
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
        dest="clini_tables",
        type=Path,
        required=True,
        action="append",
        help="Path to the clinical table. Can be specified multiple times",
    )
    parser.add_argument(
        "-s",
        "--slide-table",
        metavar="PATH",
        dest="slide_tables",
        type=Path,
        required=True,
        action="append",
        help="Path to the slide table. Can be specified multiple times",
    )
    parser.add_argument(
        "-f",
        "--feature-dir",
        metavar="PATH",
        dest="feature_dirs",
        type=Path,
        required=True,
        action="append",
        help="Path containing the slide features as `h5` files. Can be specified multiple times",
    )

    targets_parser = parser.add_mutually_exclusive_group(required=True)
    targets_parser.add_argument(
        "-t",
        "--target-label",
        metavar="LABEL",
        type=str,
        action="append",
        dest="target_labels",
        help="Target labels to train for. Can be specified multiple times",
    )
    targets_parser.add_argument(
        "--target-file",
        metavar="PATH",
        type=Path,
        help="A file containing a list of target labels, one per line.",
    )
    parser.add_argument(
        "--filter-targets",
        action="store_true",
        help="Automatically filter out nonsensical targets",
    )

    parser.add_argument(
        "--patient-col",
        metavar="COL",
        type=str,
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
    training_parser.add_argument(
        "--num-splits",
        metavar="N",
        type=int,
        default=6,
        help="Number of splits during cross-validation",
    )

    return parser


def get_splits(
    items: npt.NDArray[Any], n_splits: int = 6
) -> Iterator[Tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]]:
    splitter = KFold(n_splits=n_splits, shuffle=True)
    folds = np.array([fold for _, fold in splitter.split(items)], dtype=int)
    for test_fold, test_fold_idxs in enumerate(folds):
        val_fold = (test_fold + 1) % n_splits
        val_fold_idxs = folds[val_fold]

        train_folds = set(range(n_splits)) - {test_fold, val_fold}
        train_fold_idxs = np.concatenate(folds[list(train_folds)])

        yield (
            items[train_fold_idxs],
            items[val_fold_idxs],
            items[test_fold_idxs],
        )


def make_dataloaders(
    *,
    train_bags: Sequence[Iterable[Path]],
    train_targets: torch.Tensor,
    valid_bags: Sequence[Iterable[Path]],
    valid_targets: torch.Tensor,
    test_bags: Sequence[Iterable[Path]],
    test_targets: torch.Tensor,
    batch_size: int,
    instances_per_bag: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
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

    test_ds = BagDataset(
        bags=test_bags,
        targets=test_targets,
        instances_per_bag=instances_per_bag,
        deterministic=True,
    )
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    return train_dl, valid_dl, test_dl


if __name__ == "__main__":
    main()
