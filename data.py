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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.classification.auroc import _multilabel_auroc_compute
from torchmetrics.utilities.data import dim_zero_cat, select_topk

__all__ = ["BagDataset"]


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
    """The label of each bag."""
    instances_per_bag: int
    """The number of instances to sample."""
    deterministic: bool
    """Whether to sample deterministically.
    
    If true, `instances_per_bag` samples will be taken equidistantly from the
    bag.  Otherwise, they will be sampled randomly.
    """

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
        # too few features; pad with zeros
        pad_size = n - a.size(0)
        return torch.cat([a, torch.zeros(pad_size, *a.shape[1:])])
    elif deterministic:
        # sample equidistantly
        return a[torch.linspace(0, len(a) - 1, steps=n, dtype=torch.int)]
    else:
        # sample randomly
        idx = torch.randperm(a.size(0))[:n]
        return a[idx]
