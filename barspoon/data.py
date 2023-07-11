from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

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
    targets: Mapping[str, torch.Tensor]
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

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns features, positions, targets"""

        # Collect features from all requested slides
        feat_list, coord_list = [], []
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

                feats, coords = (
                    torch.tensor(f["feats"][:]),
                    torch.tensor(f["coords"][:]),
                )

            if self.instances_per_bag:
                feats, coords = pad_or_sample(
                    feats,
                    coords,
                    n=self.instances_per_bag,
                    deterministic=self.deterministic,
                )

            feat_list.append(feats.float())
            coord_list.append(coords.float())

        feats, coords = torch.concat(feat_list), torch.concat(coord_list)

        # We sample both on the slide as well as on the bag level
        # to ensure that each of the bags gets represented
        # Otherwise, drastically larger bags could "drown out"
        # the few instances of the smaller bags
        if self.instances_per_bag is not None:
            feats, coords = pad_or_sample(
                feats,
                coords,
                n=self.instances_per_bag,
                deterministic=self.deterministic,
            )

        return (
            feats,
            coords,
            {label: target[index] for label, target in self.targets.items()},
        )


def pad_or_sample(*xs: torch.Tensor, n: int, deterministic: bool) -> List[torch.Tensor]:
    assert (
        len(set(x.shape[0] for x in xs)) == 1
    ), "all inputs have to be of equal length"
    length = xs[0].shape[0]

    if length <= n:
        # Too few features; pad with zeros
        pad_size = n - length
        padded = [torch.cat([x, torch.zeros(pad_size, *x.shape[1:])]) for x in xs]
        return padded
    elif deterministic:
        # Sample equidistantly
        idx = torch.linspace(0, len(xs) - 1, steps=n, dtype=torch.int)
        return [x[idx] for x in xs]
    else:
        # Sample randomly
        idx = torch.randperm(length)[:n]
        return [x[idx] for x in xs]
