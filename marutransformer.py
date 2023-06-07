#!/usr/bin/env python3
# %%
import argparse
import math
import re
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clini-table", type=Path, required=True)
    parser.add_argument("--slide-table", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--target-file", type=Path, required=True)
    # parser.add_argument("--target-label", type=str, required=True)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-encoder-heads", type=int, default=8)
    parser.add_argument("--num-decoder-heads", type=int, default=8)
    # parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-encoder-layers", type=int, default=2)
    parser.add_argument("--num-decoder-layers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--dim-feedforward", type=int, default=2048)
    parser.add_argument("--instances-per-bag", type=int, default=2**12)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.classification.auroc import _multilabel_auroc_compute
from torchmetrics.utilities.data import dim_zero_cat, select_topk

torch.set_float32_matmul_precision("medium")


class ParallelLinear(nn.Module):
    """Parallelly apply multiple linear layers."""

    def __init__(self, in_features: int, out_features: int, n_parallel: int):
        super().__init__()
        self.in_features, self.out_features, self.n_parallel = (
            in_features,
            out_features,
            n_parallel,
        )
        self.weight = nn.Parameter(torch.empty((n_parallel, in_features, out_features)))
        self.bias = nn.Parameter(torch.empty((n_parallel, out_features)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Adapted from torch.nn.Linear
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.out_features)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor):
        assert x.ndim in [2, 3], (
            "ParallelLinear is only defined for inputs of shape "
            "(n_parallel, in_features) and (batch_size, n_parallel, in_features)"
        )
        return (x.unsqueeze(-2) @ self.weight).squeeze(-2) + self.bias

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, n_parallel={self.n_parallel}"


@dataclass
class BagDataset(Dataset):
    """A dataset of bags of instances."""

    bags: Sequence[Iterable[Path]]
    """The `.h5` files containing the bags.

    Each bag consists of the features taken from one or multiple h5 files.
    Each of the h5 files needs to have a dataset called `feats` of shape N x F,
    where N is the number of instances and F the number of features per instance.
    """
    labels: torch.Tensor
    instances_per_bag: int
    deterministic: bool

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # collect all the features
        feat_list = []
        for bag_file in self.bags[index]:
            with h5py.File(bag_file, "r") as f:
                feat_list.append(
                    pad_or_sample(
                        torch.from_numpy(f["feats"][:]),
                        n=self.instances_per_bag,
                        deterministic=self.deterministic,
                    )
                )
        feats = pad_or_sample(
            torch.concat(feat_list).float(), 4096, deterministic=self.deterministic
        )

        return feats, self.labels[index]


def pad_or_sample(a: torch.Tensor, n: int, deterministic: bool):
    if a.size(0) <= n:
        pad_size = n - a.size(0)
        return torch.cat([a, torch.zeros(pad_size, *a.shape[1:])])
    elif deterministic:
        return a[torch.linspace(0, len(a) - 1, steps=n, dtype=int)]
    else:
        idx = torch.randperm(a.size(0))[:n]
        return a[idx]


class TopKMultilabelAUROC(torchmetrics.classification.MultilabelPrecisionRecallCurve):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        num_labels: int,
        topk: int,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    ) -> None:
        super().__init__(num_labels=num_labels, validate_args=False)
        self.topk = topk
        self.average = average

    def compute(self) -> torch.Tensor:
        state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)]
        individual_aurocs = _multilabel_auroc_compute(
            state, self.num_labels, average="none", thresholds=None
        )
        topk_idx = select_topk(individual_aurocs, self.topk, dim=0).bool()

        state = [
            dim_zero_cat(self.preds)[:, topk_idx],
            dim_zero_cat(self.target)[:, topk_idx],
        ]
        return _multilabel_auroc_compute(
            state, self.topk, average=self.average, thresholds=None
        )


class VisionTransformer(nn.Module):
    def __init__(
        self,
        d_features: int,
        n_targets: int,
        *,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
    ) -> None:
        super().__init__()

        # one class token per output class
        self.class_tokens = nn.Parameter(torch.rand(n_targets, d_model))

        self.projector = nn.Sequential(nn.Linear(d_features, d_model), nn.ReLU())

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder_stack = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=encoder_norm,
        )
        self.heads = ParallelLinear(
            in_features=d_model, out_features=1, n_parallel=n_targets
        )

    def forward(self, bag):
        batch_size, _, _ = bag.shape
        projected = self.projector(bag)  # shape: [bs, seq_len, d_model]

        # prepend class tokens
        out_features, d_model = self.class_tokens.shape
        with_class_tokens = torch.cat(
            [
                self.class_tokens[None, :, :].expand(batch_size, out_features, d_model),
                projected,
            ],
            dim=1,
        )
        encoded = self.encoder_stack(with_class_tokens)

        # apply the corresponding head to each class token
        logits = self.heads(encoded[:, :out_features, :]).squeeze(-1)

        return logits


class EncoderDecoderTransformer(nn.Module):
    def __init__(
        self,
        d_features: int,
        n_targets: int,
        *,
        d_model: int = 512,
        num_encoder_heads: int = 8,
        num_decoder_heads: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 2048,
    ) -> None:
        """
             ┏━━┓      ┏━━┓           ┏━━┓
        t ─▶┃E0┠──┬─▶┃E1┠──┬──···─▶┃En┠──┐
             ┗━━┛  │   ┗━━┛  │        ┗━━┛  │
                   ▼         ▼              ▼
                 ┏━━┓      ┏━━┓           ┏━━┓   ┏━━┓
        c ─────▶┃D0┠────▶┃D1┠─···─────▶┃Dn┠─▶┃FC┠─▶ s
                 ┗━━┛      ┗━━┛           ┗━━┛   ┗━━┛
        """
        super().__init__()

        # one class token per output class
        self.class_tokens = nn.Parameter(torch.rand(n_targets, d_model))

        self.projector = nn.Sequential(nn.Linear(d_features, d_model), nn.ReLU())

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_encoder_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_decoder_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        self.heads = ParallelLinear(
            in_features=d_model, out_features=1, n_parallel=n_targets
        )

    def forward(self, tile_tokens):
        batch_size, _, _ = tile_tokens.shape

        tile_tokens = self.projector(tile_tokens)  # shape: [bs, seq_len, d_model]
        tile_tokens = self.transformer_encoder(tile_tokens)

        class_tokens = self.class_tokens.expand(batch_size, -1, -1)
        class_tokens = self.transformer_decoder(tgt=class_tokens, memory=tile_tokens)

        # apply the corresponding head to each class token
        logits = self.heads(class_tokens).squeeze(-1)

        return logits


class BarspoonTransformer(nn.Module):
    def __init__(
        self,
        d_features: int,
        n_targets: int,
        *,
        d_model: int = 512,
        num_encoder_heads: int = 8,
        num_decoder_heads: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
    ) -> None:
        """
             ┏━━┓      ┏━━┓           ┏━━┓
        t ─▶┃E0┠──┬─▶┃E1┠──┬──···─▶┃En┠──┐
             ┗━━┛  │   ┗━━┛  │        ┗━━┛  │
                   ▼         ▼              ▼
                 ┏━━┓      ┏━━┓           ┏━━┓   ┏━━┓
        c ─────▶┃D0┠────▶┃D1┠─···─────▶┃Dn┠─▶┃FC┠─▶ s
                 ┗━━┛      ┗━━┛           ┗━━┛   ┗━━┛
        """
        super().__init__()

        # one class token per output class
        self.class_tokens = nn.Parameter(torch.rand(n_targets, d_model))

        self.projector = nn.Sequential(nn.Linear(d_features, d_model), nn.ReLU())

        self.encoder_decoder_pairs = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "encoder": nn.TransformerEncoderLayer(
                            d_model=d_model,
                            nhead=num_encoder_heads,
                            dim_feedforward=dim_feedforward,
                            batch_first=True,
                            norm_first=True,
                        ),
                        "decoder": nn.TransformerDecoderLayer(
                            d_model,
                            num_decoder_heads,
                            dim_feedforward=dim_feedforward,
                            batch_first=True,
                            norm_first=True,
                        ),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        # TODO maybe we need a norm???
        self.heads = ParallelLinear(
            in_features=d_model, out_features=1, n_parallel=n_targets
        )

    def forward(self, tile_tokens):
        batch_size, _, _ = tile_tokens.shape

        tile_tokens = self.projector(tile_tokens)  # shape: [bs, seq_len, d_model]
        class_tokens = self.class_tokens.expand(batch_size, -1, -1)

        for layer in self.encoder_decoder_pairs:
            tile_tokens = layer["encoder"](tile_tokens)
            class_tokens = layer["decoder"](tgt=class_tokens, memory=tile_tokens)

        # apply the corresponding head to each class token
        logits = self.heads(class_tokens).squeeze(-1)

        return logits


class LitMilClassificationMixin(pl.LightningModule):
    def __init__(
        self,
        *,
        target_labels: Sequence[str],
        pos_weight: Optional[torch.Tensor],
        # other hparams
        learning_rate: float = 1e-4,
        **hparams: Any,
    ) -> None:
        super().__init__()
        _ = hparams  # so we don't get unused parameter warnings

        self.learning_rate = learning_rate
        n_targets = len(target_labels)

        # use the same metrics for training, validation and testing
        global_metrics = torchmetrics.MetricCollection(
            [
                TopKMultilabelAUROC(
                    num_labels=n_targets, topk=max(int(n_targets * 0.2), 1)
                )
            ]
        )
        target_aurocs = torchmetrics.MetricCollection(
            {
                sanatize(target_label): torchmetrics.AUROC(task="binary")
                for target_label in target_labels
            }
        )
        for step_name in ["train", "val", "test"]:
            setattr(
                self,
                f"{step_name}_global_metrics",
                global_metrics.clone(prefix=f"{step_name}_"),
            )
            setattr(
                self,
                f"{step_name}_target_aurocs",
                target_aurocs.clone(prefix=f"{step_name}_"),
            )

        self.target_labels = target_labels
        self.pos_weight = pos_weight

        self.save_hyperparameters()

    def step(self, batch, step_name=None):
        bags, targets = batch
        logits = self(bags)
        # BCE ignoring the positions we don't have target labels for
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight.type_as(targets),
            reduction="none",
        ).nansum() / len(self.target_labels)
        if step_name:
            # update global metrics
            global_metrics = getattr(self, f"{step_name}_global_metrics")
            global_metrics.update(logits, targets.long())
            self.log_dict(
                global_metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                f"{step_name}_loss", loss, on_step=True, on_epoch=True, sync_dist=True
            )

            # update target-wise metrics
            for target_label, target_logits, target_ys in zip(
                self.target_labels,
                logits.permute(-1, -2),
                targets.permute(-1, -2),
                # strict=True,  # python3.9 hates it
            ):
                target_auroc = getattr(self, f"{step_name}_target_aurocs")[
                    sanatize(target_label)
                ]
                target_auroc.update(target_logits, target_ys)
                self.log(
                    f"{step_name}_{target_label}_auroc",
                    target_auroc,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, step_name="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, step_name="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, step_name="test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        bag, targets = batch
        logits = self(bag)
        return torch.sigmoid(logits), targets

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class LitEncoderDecoderTransformer(LitMilClassificationMixin):
    def __init__(
        self,
        *,
        d_features: int,
        target_labels: Sequence[str],
        pos_weight: Optional[torch.Tensor],
        # model parameters
        d_model: int,  # = 512,
        num_encoder_heads: int,  # = 8,
        num_decoder_heads: int,  # = 8,
        num_encoder_layers: int,  # = 2,
        num_decoder_layers: int,  # = 2,
        dim_feedforward: int,  # = 2048,
        # other hparams
        learning_rate: float,  # = 1e-4,
        **hparams: Any,
    ) -> None:
        super().__init__(
            target_labels=target_labels,
            pos_weight=pos_weight,
            learning_rate=learning_rate,
        )
        _ = hparams  # so we don't get unused parameter warnings

        self.model = EncoderDecoderTransformer(
            d_features=d_features,
            n_targets=len(target_labels),
            d_model=d_model,
            num_encoder_heads=num_encoder_heads,
            num_decoder_heads=num_decoder_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
        )

        self.save_hyperparameters()

    def forward(self, tile_tokens):
        return self.model(tile_tokens)


class LitBarspoonTransformer(LitMilClassificationMixin):
    def __init__(
        self,
        *,
        d_features: int,
        target_labels: Sequence[str],
        pos_weight: Optional[torch.Tensor],
        # model parameters
        d_model: int,  # = 512,
        num_encoder_heads: int,  # = 8,
        num_decoder_heads: int,  # = 8,
        num_layers: int,  # = 2,
        dim_feedforward: int,  # = 2048,
        # other hparams
        learning_rate: float,  # = 1e-4,
        **hparams: Any,
    ) -> None:
        super().__init__(
            target_labels=target_labels,
            pos_weight=pos_weight,
            learning_rate=learning_rate,
        )
        _ = hparams  # so we don't get unused parameter warnings

        self.model = BarspoonTransformer(
            d_features=d_features,
            n_targets=len(target_labels),
            d_model=d_model,
            num_encoder_heads=num_encoder_heads,
            num_decoder_heads=num_decoder_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
        )

        self.save_hyperparameters()

    def forward(self, tile_tokens):
        return self.model(tile_tokens)


class LitVisionTransformer(LitMilClassificationMixin):
    def __init__(
        self,
        *,
        d_features: int,
        target_labels: Sequence[str],
        pos_weight: Optional[torch.Tensor],
        # model parameters
        d_model: int,  # = 512,
        num_heads: int,  # = 8,
        num_layers: int,  # = 2,
        dim_feedforward: int,  # = 2048,
        # other hparams
        learning_rate: float,  # = 1e-4,
        **hparams: Any,
    ) -> None:
        super().__init__(
            target_labels=target_labels,
            pos_weight=pos_weight,
            learning_rate=learning_rate,
        )
        _ = hparams  # so we don't get unused parameter warnings

        self.model = VisionTransformer(
            d_features=d_features,
            n_targets=len(target_labels),
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
        )

        self.save_hyperparameters()

    def forward(self, tile_tokens):
        return self.model(tile_tokens)


def sanatize(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", x)


def read_table(path: Path, dtype=str) -> pd.DataFrame:
    if Path(path).suffix == ".csv":
        return pd.read_csv(path, dtype=dtype)
    else:
        return pd.read_excel(path, dtype=dtype)


def get_splits(X, n_splits: int = 6):
    # splitter = KFold(n_splits=n_splits, shuffle=True)
    # folds = np.array([fold for _, fold in splitter.split(X=X)], dtype=object)
    # for test_fold, test_fold_idxs in enumerate(folds):
    #     val_fold = (test_fold + 1) % n_splits
    #     val_fold_idxs = folds[val_fold]

    #     train_folds = set(range(n_splits)) - {test_fold, val_fold}
    #     train_fold_idxs = np.concatenate(folds[list(train_folds)])

    #     yield (
    #         train_fold_idxs.astype(int),
    #         val_fold_idxs.astype(int),
    #         test_fold_idxs.astype(int),
    #     )

    folds = [
        cohort_df.index[cohort_df.PATIENT.isin(test_patients)].values
        for test_patients in np.load("folds.npy")
    ]
    folds = np.array(folds)
    for test_fold, test_fold_idxs in enumerate(folds):
        val_fold = (test_fold + 1) % n_splits
        val_fold_idxs = folds[val_fold]

        train_folds = set(range(n_splits)) - {test_fold, val_fold}
        train_fold_idxs = np.concatenate(folds[list(train_folds)])

        yield (
            train_fold_idxs.astype(int),
            val_fold_idxs.astype(int),
            test_fold_idxs.astype(int),
        )


# %%
import random

# %%

if __name__ == "__main__":
    # vision
    # args.d_model = round(2 ** random.uniform(5, 11)) // 8 * 8
    # args.num_heads = random.choice([2, 4, 8])
    # args.num_layers = random.randint(1, 6)
    # args.dim_feedforward = round(2 ** random.uniform(8, 12))

    # barspoon
    # args.num_encoder_heads = random.choice([2, 4, 8])
    # args.num_decoder_heads = random.choice([2, 4, 8])
    # args.num_layers = random.randint(1, 6)
    # args.d_model = round(2 ** random.uniform(5, 11)) // 8 * 8
    # args.dim_feedforward = round(2 ** random.uniform(8, 12))

    # other
    # args.instances_per_bag = round(2 ** random.uniform(9, 12))
    # args.learning_rate = 10 ** (random.uniform(-3, -5))
    batch_size = 4

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

    with open(args.target_file) as targets_file:
        target_labels = np.array(
            [target_label.strip() for target_label in targets_file]
        )

    target_labels = target_labels[cohort_df[target_labels].nunique(dropna=True) == 2]
    target_labels = (
        cohort_df[target_labels]
        .select_dtypes(["int16", "int32", "int64", "float16", "float32", "float64"])
        .columns.values
    )
    # target_labels = [args.target_label]

    targets = torch.Tensor(cohort_df[target_labels].apply(pd.to_numeric).values)
    bags = cohort_df.slide_path.values
    pos_samples = targets.nansum(dim=0)
    neg_samples = (1 - targets).nansum(dim=0)
    pos_weight = neg_samples / pos_samples

    configuration_str = "-".join(
        f"{k}={v}" for k, v in sorted(vars(args).items()) if isinstance(v, (int, float))
    )

    for fold_no, (train_idx, valid_idx, test_idx) in enumerate(get_splits(X=targets)):
        run_dir = args.output_dir / configuration_str / f"{fold_no=}"
        print(f"{run_dir=}")
        if (run_dir / "done").exists():
            # already done... skip
            continue
        elif run_dir.exists():
            # failed; start over
            shutil.rmtree(run_dir)

        train_ds = BagDataset(
            bags[train_idx],
            targets[train_idx],
            instances_per_bag=args.instances_per_bag,
            deterministic=False,
        )
        train_dl = DataLoader(
            train_ds, batch_size=batch_size, num_workers=8, shuffle=True
        )

        valid_ds = BagDataset(
            bags[valid_idx],
            targets[valid_idx],
            instances_per_bag=args.instances_per_bag,
            deterministic=True,
        )
        valid_dl = DataLoader(valid_ds, batch_size=batch_size, num_workers=8)

        test_ds = BagDataset(
            bags[test_idx],
            targets[test_idx],
            instances_per_bag=2**12,
            deterministic=True,
        )
        test_dl = DataLoader(test_ds, batch_size=1, num_workers=8)

        example_bags, _ = next(iter(train_dl))
        d_features = example_bags.size(-1)

        model = LitEncoderDecoderTransformer(
            d_features=d_features,
            n_targets=len(target_labels),
            pos_weight=pos_weight,
            # other hparams
            target_labels=list(target_labels),
            train_patients=list(cohort_df.loc[train_idx].PATIENT),
            valid_patients=list(cohort_df.loc[valid_idx].PATIENT),
            test_patients=list(cohort_df.loc[test_idx].PATIENT),
            **vars(args),
        )

        trainer = pl.Trainer(
            default_root_dir=run_dir,
            callbacks=[
                EarlyStopping(
                    monitor="val_TopKMultilabelAUROC", mode="max", patience=16
                ),
                ModelCheckpoint(
                    monitor="val_TopKMultilabelAUROC",
                    mode="max",
                    filename="checkpoint-{epoch:02d}-{val_TopKMultilabelAUROC:0.3f}",
                ),
            ],
            accelerator="auto",
            accumulate_grad_batches=32 // batch_size,
            gradient_clip_val=0.5,
            logger=CSVLogger(save_dir=run_dir),
        )

        trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=valid_dl)
        trainer.test(model=model, dataloaders=test_dl)

        with open(run_dir / "done", "w"):
            pass
