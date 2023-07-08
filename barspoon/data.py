from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import h5py
import torch
from torch.utils.data import Dataset

__all__ = ["BagDataset"]


@dataclass
class BagDataset(Dataset):
    """A dataset of bags of instances"""

    bags: Sequence[Iterable[Path]]
    """The `.h5` files containing the bags

    Each bag consists of the features taken from one or multiple h5 files.
    Each of the h5 files needs to have a dataset called `feats` of shape N x F,
    where N is the number of instances and F the number of features per instance.
    """
    targets: torch.Tensor
    """The label of each bag"""
    instances_per_bag: Optional[int]
    """The number of instances to sample, or all samples if None"""
    deterministic: bool = True
    """Whether to sample deterministically
    
    If true, `instances_per_bag` samples will be taken equidistantly from the
    bag.  Otherwise, they will be sampled randomly.
    """

    extractor: Optional[str] = None
    """Feature extractor the features got extracted with

    Set on first encountered feature, if not set manually.  Will raise an error
    during runtime if features extracted with different feature extractors are
    encountered in the dataset.
    """

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns features, targets"""

        # Collect features from all requested slides
        feat_list = []
        for bag_file in self.bags[index]:
            with h5py.File(bag_file, "r") as f:
                # Ensure all features are created with the same feature extractor
                this_slides_extractor = f.attrs.get("extractor")
                if self.extractor is None:
                    self.extractor = this_slides_extractor
                assert this_slides_extractor == self.extractor, (
                    "all features have to be extracted with the same feature extractor! "
                    f"{bag_file} has been extracted with {this_slides_extractor}, "
                    f"expected {self.extractor}"
                )

                feat_list.append(
                    torch.from_numpy(f["feats"][:])
                    if self.instances_per_bag is None
                    else pad_or_sample(
                        torch.from_numpy(f["feats"][:]),
                        n=self.instances_per_bag,
                        deterministic=self.deterministic,
                    )
                )
        feats = torch.concat(feat_list).float()

        if self.instances_per_bag is not None:
            feats = pad_or_sample(
                feats, self.instances_per_bag, deterministic=self.deterministic
            )

        return feats, self.targets[index]


def pad_or_sample(a: torch.Tensor, n: int, deterministic: bool):
    if a.size(0) <= n:
        # Too few features; pad with zeros
        pad_size = n - a.size(0)
        return torch.cat([a, torch.zeros(pad_size, *a.shape[1:])])
    elif deterministic:
        # Sample equidistantly
        return a[torch.linspace(0, len(a) - 1, steps=n, dtype=torch.int)]
    else:
        # Sample randomly
        idx = torch.randperm(a.size(0))[:n]
        return a[idx]
