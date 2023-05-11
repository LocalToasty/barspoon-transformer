#!/usr/bin/env python3
# %%
import math
import re
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Tuple
import sys

import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.classification.auroc import _multilabel_auroc_compute
from torchmetrics.utilities.data import dim_zero_cat, select_topk


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


class ViTransformer(pl.LightningModule):
    def __init__(
        self,
        d_features: int,
        n_targets: int,
        pos_weight: Optional[torch.Tensor],
        *,
        learning_rate: float = 1e-4,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        **hparams: Any,
    ) -> None:
        super().__init__()
        _ = hparams

        self.learning_rate = learning_rate

        # one class token per output class
        self.class_tokens = nn.Parameter(torch.rand(n_targets, d_model))

        self.projector = nn.Sequential(nn.Linear(d_features, d_model), nn.ReLU())

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, batch_first=True, norm_first=True
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder_stack = nn.TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            norm=encoder_norm,
        )
        self.heads = ParallelLinear(
            in_features=d_model, out_features=1, n_parallel=n_targets
        )

        metrics = torchmetrics.MetricCollection(
            [
                TopKMultilabelAUROC(
                    num_labels=n_targets, topk=max(int(n_targets * 0.2), 1)
                )
            ]
        )
        for step in ["train", "val", "test"]:
            setattr(self, f"{step}_global_metrics", metrics.clone(prefix=f"{step}_"))

        self.pos_weight = pos_weight

        self.save_hyperparameters()

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

    def step(self, batch, step_name=None):
        bag, target = batch
        logits = self(bag)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=self.pos_weight.type_as(target),
            reduction="none",
        ).nansum()
        if step_name:
            metrics = getattr(self, f"{step_name}_metrics")
            metrics.update(logits, target.long())
            self.log_dict(
                metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
            )

            self.log(
                f"{step_name}_loss", loss, on_step=True, on_epoch=True, sync_dist=True
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


class BarspoonTransformer(pl.LightningModule):
    """
        ┏━━┓     ┏━━┓          ┏━━┓
    t ─▶┃E0┠──┬─▶┃E1┠──┬──···─▶┃En┠──┐
        ┗━━┛  │  ┗━━┛  │       ┗━━┛  │
              ▼        ▼             ▼
            ┏━━┓     ┏━━┓          ┏━━┓  ┏━━┓
    c ─────▶┃D0┠────▶┃D1┠─···─────▶┃Dn┠─▶┃FC┠─▶ s
            ┗━━┛     ┗━━┛          ┗━━┛  ┗━━┛
    """

    def __init__(
        self,
        d_features: int,
        n_targets: int,
        *,
        target_labels: Sequence[str],
        pos_weight: Optional[torch.Tensor],
        # model parameters
        d_model: int = 512,
        n_encoder_heads: int = 8,
        n_decoder_heads: int = 8,
        n_layers: int = 2,
        dim_feedforward: int = 2048,
        # other hparams
        learning_rate: float = 1e-4,
        **hparams: Any,
    ) -> None:
        super().__init__()
        _ = hparams  # so we don't get unused parameter warnings

        self.learning_rate = learning_rate

        # one class token per output class
        self.class_tokens = nn.Parameter(torch.rand(n_targets, d_model))

        self.projector = nn.Sequential(nn.Linear(d_features, d_model), nn.ReLU())

        self.encoder_decoder_pairs = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "encoder": nn.TransformerEncoderLayer(
                            d_model,
                            n_encoder_heads,
                            dim_feedforward=dim_feedforward,
                            batch_first=True,
                            norm_first=True,
                        ),
                        "decoder": nn.TransformerDecoderLayer(
                            d_model,
                            n_decoder_heads,
                            dim_feedforward=dim_feedforward,
                            batch_first=True,
                            norm_first=True,
                        ),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        # TODO maybe we need a norm???
        self.heads = ParallelLinear(
            in_features=d_model, out_features=1, n_parallel=n_targets
        )

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

    def step(self, batch, step_name=None):
        bags, targets = batch
        logits = self(bags)
        # BCE ignoring the positions we don't have target labels for
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight.type_as(targets),
            reduction="none",
        ).nansum()
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


def sanatize(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", x)


def read_table(path: Path, dtype=str) -> pd.DataFrame:
    if Path(path).suffix == ".csv":
        return pd.read_csv(path, dtype=dtype)
    else:
        return pd.read_excel(path, dtype=dtype)


batch_size = 1
clini_df = read_table(
    "/mnt/bulk/mvantreeck/gecco/gecco-multilabel/CLINI_Gecco_All_v6_MG.xlsx",
    dtype={"PATIENT": str},
)
slide_df = read_table(
    "/mnt/bulk/mvantreeck/gecco/gecco-multilabel/SLIDE_GECCO_IWHS.csv"
)[["PATIENT", "FILENAME"]]
df = clini_df.merge(slide_df, on="PATIENT")
assert not df.empty, "no overlap between clini and slide table."

feature_dir = Path("/mnt/bulk/mvantreeck/gecco/gecco-multilabel/IWHS")
# remove slides we don't have
h5s = set(feature_dir.glob("*.h5"))
assert h5s, f"no features found in {feature_dir}!"
h5_df = pd.DataFrame(h5s, columns=["slide_path"])
h5_df["FILENAME"] = h5_df.slide_path.map(lambda p: p.stem)
cohort_df = df.merge(h5_df, on="FILENAME").reset_index()
# reduce to one row per patient with list of slides in `df['slide_path']`
patient_df = cohort_df.groupby("PATIENT").first().drop(columns="slide_path")
patient_slides = cohort_df.groupby("PATIENT").slide_path.apply(list)
cohort_df = patient_df.merge(
    patient_slides, left_on="PATIENT", right_index=True
).reset_index()

with open("/mnt/bulk/mvantreeck/gecco/gecco-multilabel/targets_1.txt") as targets_file:
    target_labels = np.array([target_label.strip() for target_label in targets_file])

target_labels = target_labels[cohort_df[target_labels].nunique(dropna=True) == 2]
target_labels = (
    cohort_df[target_labels]
    .select_dtypes(["int16", "int32", "int64", "float16", "float32", "float64"])
    .columns.values
)

targets = torch.Tensor(cohort_df[target_labels].apply(pd.to_numeric).values)
bags = cohort_df.slide_path.values
pos_samples = targets.nansum(dim=0)
neg_samples = (1 - targets).nansum(dim=0)
pos_weight = neg_samples / pos_samples


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
root_dir = Path(f"/mnt/bulk/mvantreeck/gecco/interleaved/")
hparam_grid = [
    {
        "n_encoder_heads": n_encoder_heads,
        "n_decoder_heads": n_decoder_heads,
        "n_layers": n_layers,
        "dim_feedforward": dim_feedforward,
        "d_model": d_model,
    }
    for n_encoder_heads in [8, 16]
    for n_decoder_heads in [8, 16]
    for n_layers in [1, 2, 4, 6]
    for dim_feedforward in [1024, 2048]
    for d_model in [256, 512]
    if not all([(root_dir/f"{n_layers=}-{n_encoder_heads=}-{n_decoder_heads=}-{d_model=}-{dim_feedforward=}/{fold_no=}/done").exists()
        for fold_no in range(6)])
]
# %%
array_id = int(sys.argv[1])
print(f"job {array_id:3}/{len(hparam_grid)}")
# %%
hparams = hparam_grid[array_id]
n_encoder_heads = hparams["n_encoder_heads"]
n_decoder_heads = hparams["n_decoder_heads"]
n_layers = hparams["n_layers"]
d_model = hparams["d_model"]
dim_feedforward = hparams["dim_feedforward"]

for fold_no, (train_idx, valid_idx, test_idx) in enumerate(get_splits(X=targets)):
    print(f"{hparams=}, {fold_no=}")
    run_dir = root_dir/f"{n_layers=}-{n_encoder_heads=}-{n_decoder_heads=}-{d_model=}-{dim_feedforward=}/{fold_no=}"
    if (run_dir / "done").exists():
        # already done... skip
        continue
    elif run_dir.exists():
        # failed; start over
        shutil.rmtree(run_dir)

    train_ds = BagDataset(
        bags[train_idx],
        targets[train_idx],
        instances_per_bag=2**10,
        deterministic=False,
    )
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, num_workers=8, shuffle=True
    )

    valid_ds = BagDataset(
        bags[valid_idx],
        targets[valid_idx],
        instances_per_bag=2**10,
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

    model = BarspoonTransformer(
        d_features=d_features,
        n_targets=len(target_labels),
        pos_weight=pos_weight,
        # other hparams
        target_labels=list(target_labels),
        train_patients=list(cohort_df.loc[train_idx].PATIENT),
        valid_patients=list(cohort_df.loc[valid_idx].PATIENT),
        test_patients=list(cohort_df.loc[test_idx].PATIENT),
        n_encoder_heads=n_encoder_heads,
        n_decoder_heads=n_decoder_heads,
        n_layers=n_layers,
        d_model=d_model,
        dim_feedforward=dim_feedforward,
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
    )

    trainer.fit(
        model=model, train_dataloaders=train_dl, val_dataloaders=valid_dl
    )
    trainer.test(model=model, dataloaders=test_dl)

    with open(run_dir / "done-new", "w"):
        pass
# %%

# from sklearn.metrics import roc_auc_score

# def get_aurocs(bags, targets):
#     predict_ds = BagDataset(
#         bags, targets, instances_per_bag=2**12, deterministic=True
#     )
#     predict_dl = DataLoader(predict_ds, batch_size=1, num_workers=8)
#     trainer = pl.Trainer(
#         default_root_dir=f"ipsum",
#         max_epochs=1,
#         accelerator="auto",
#         devices=1,
#     )
#     x = trainer.predict(model, predict_dl)

#     scores = torch.stack([scores.squeeze(-2) for scores, _ in x])
#     targets = torch.stack([targets.squeeze(-2) for _, targets in x])

#     predict_aurocs = {}
#     for target_label, t, s in zip(target_labels, targets.transpose(0,1), scores.transpose(0,1), strict=True):
#         if (t == 1).all() or (t == 0).all():
#             continue
#         predict_aurocs[target_label] = roc_auc_score(t, s)

#     return predict_aurocs

# # %%
# aurocs = {}
# for fold_no in range(6):
#     model_path = next(Path(f"lorem-{fold_no=}/lightning_logs/version_0/checkpoints").glob("checkpoint-*.ckpt"))
#     model = BarspoonTransformer.load_from_checkpoint(model_path)
#     for part in ["valid_patients", "test_patients"]:
#         test_df = cohort_df[cohort_df.PATIENT.isin(model.hparams[part])]
#         targets = torch.Tensor(test_df[model.hparams["target_labels"]].apply(pd.to_numeric).values)
#         bags = test_df.slide_path.values

#         aurocs[(fold_no, part)] = get_aurocs(bags, targets)
# # %%
# import matplotlib.pyplot as plt
# import numpy as np

# fig, ax = plt.subplots(subplot_kw={"aspect": "equal"})
# ax.plot([.3,1], [.3,1], 'r--')
# for target_label in target_labels:
#     test_scores = [aurocs[fold_no, "test_patients"].get(target_label) for fold_no in range(6)]
#     marugoto_scores = []
#     try:
#         for fold_no in range(5):
#             df = pd.read_csv(f"/mnt/bulk/mvantreeck/gecco/gecco-marugoto-results/{target_label}/fold-{fold_no}/patient-preds.csv")
#             marugoto_scores.append(roc_auc_score(df[target_label], df[f"{target_label}_1"]))
#     except Exception:
#         continue
#     if None in (set(marugoto_scores) | set(test_scores)):
#         continue
#     # if min(valid_scores) < .6:
#     #     continue

#     plt.scatter([np.mean(marugoto_scores)], [np.mean(test_scores)], marker='.')

# ax.set_aspect("equal")
# plt.title("Median AUROC for GECCO IWHS Crossval")
# plt.xlabel("marugoto")
# plt.ylabel("multilabel")
# plt.hlines([.5], .3, 1, ['r'], linestyles='dotted')
# plt.vlines([.5], .3, 1, ['r'], linestyles='dotted')
# # %%
# %%
