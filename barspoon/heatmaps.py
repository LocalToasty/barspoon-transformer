"""Generate heatmaps for barspoon transformer models

# Heatmap Computation

We use gradcam to generate the heatmaps.  This has the advantage of being
relatively model-agnostic and thus, we hope, more "objective".  It furthermore
simplifies implementation, as most of the primitives needed for gradcam come
included in pytorch.

We calculate the gradcam g of a slide consisting of tile feature vectors x as
follows:

```asciimath
g(x) = sum_i dy/dx_i * x_i
```

where `x_i` is the `i`-th tile's feature vector and `y` is the output we want to
compute a gradcam heatmap for.

The intuition about this is as follows:  `dy/dx_i` tells us how sensitive the
network is to changes in the feature `x_i`.  Since `x_i` which are large in
magnitude affect the output of the network stronger small ones, `dy/dx_i * x_i`
gives us an overall measure of how strongly `x_i` contributed to `y`.  Positive
`dy/dx_i * x_i` imply that this feature positively affected `y`, while negative
ones point towards this feature reducing the value of `y` in our result.  By
summing over all `x_i`, we get the overall contribution of the tile to the final
result `y`.

# "Attention" vs "Contribution"

We output two kinds of heatmaps:

 1. The Attention Map shows which part of the input image had an effect on the
    final result.  It is computed as `|g(x)|`, i.e. the absolute value of the
    gradcam scores.
 2. The Contribution Map is simply `g(x)`, with tiles positively contributing to
    the result being displayed as red ("hot") and those negatively contributing
    to it as blue ("cold").  Tiles with less contribution are also rendered in
    see-through optic, such that if rendered on top of a white background their
    brightness is roughly equivalent to their contribtion.
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from packaging.specifiers import SpecifierSet
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from tqdm import tqdm

from barspoon.model import LitEncDecTransformer
from barspoon.utils import make_dataset_df


def main():
    parser = make_argument_parser()
    args = parser.parse_args()

    # Load model and ensure its version is compatible
    # We do all computations in bfloat16, as it needs way less VRAM and performs
    # virtually identical to float32 in inference tasks.
    model = LitEncDecTransformer.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path
    ).to(device=args.device, dtype=torch.bfloat16)
    name, version = model.hparams.get("version", "undefined 0").split(" ")
    if not (
        name == "barspoon-transformer"
        and (spec := SpecifierSet(">=3.0,<4")).contains(version)
    ):
        raise ValueError(
            f"model not compatible. Found {name} {version}, expected barspoon-transformer {spec}",
            model.hparams["version"],
        )

    target_labels = model.hparams["target_labels"]

    # Load dataset, grouped by filename
    dataset_df = make_dataset_df(
        feature_dirs=args.feature_dirs,
        target_labels=target_labels,
    )

    for slides in tqdm(dataset_df.path):
        assert (
            len(slides) == 1
        ), "there should only be one slide per item after grouping by slidename"
        h5_path = slides[0]

        # Load features
        with h5py.File(h5_path) as h5_file:
            feats = torch.tensor(
                h5_file["feats"][:], dtype=torch.bfloat16, device=args.device
            )
            coords = torch.tensor(h5_file["coords"][:])
            xs = np.sort(np.unique(coords[:, 0]))
            stride = np.min(xs[1:] - xs[:-1])

        categories = model.hparams["categories"]

        # Generate the gradcams
        # Skip the slide if we are out of memory
        try:
            gradcams = compute_gradcams(
                model=model,
                feats=feats,
                coords=coords,
                stride=stride,
                categories=categories,
                batch_size=args.batch_size,
            )
        except torch.cuda.OutOfMemoryError as oom_error:  # type: ignore
            logging.error(f"error while processing {slides[0]}: {oom_error})")
            continue

        # The mask is 1 wherever we have a feature
        mask = vals_to_im(np.ones((len(coords), 1)), coords.numpy(), stride)[0]
        # The gradcam entry with the overall highest magnitude in the slide.
        # Used for scaling.
        slide_abs_max = abs(
            # We first have to re-flatten all the gradcam heatmaps
            # TODO maybe it's easier to just pass the gradcam images around in
            # their stacked form?
            np.stack(
                [
                    list(g)
                    for categories in gradcams.values()
                    for g in categories.values()
                ]
            )
        ).max()

        # Save all the heatmaps
        (outdir := args.output_dir / h5_path.stem).mkdir(exist_ok=True, parents=True)
        for target_label, categories in gradcams.items():
            gradcam_im = plt.get_cmap("magma")(
                abs(np.stack(categories.values())).mean(0) / slide_abs_max
            )
            gradcam_im[:, :, -1] = mask
            Image.fromarray(
                np.uint8(255 * gradcam_im),
                "RGBA",
            ).resize(
                # Save the image x8 because that tends to be easier to work with
                np.array(mask.shape)[::-1] * 8,
                resample=Image.Resampling.NEAREST,
            ).save(
                outdir / f"gradcam_absolute_{target_label}.png",
                pnginfo=make_metadata(
                    # Metadata format semver
                    # Update minor version when adding fields,
                    # Major when removing fields / changing semantics
                    version="barspoon-absolute-gradcam 1.0",
                    filename=h5_path.stem,
                    stride=str(stride / 8),
                    target_label=target_label,
                    scale=f"{slide_abs_max}",
                ),
            )

            for category, cat_gradcam in categories.items():
                # for category, cat_gradcam in zip(cs, gradcams[left:right], strict=True):
                # Both the class-indicating map and the alpha map contain
                # information on the "importance" of a region, where the
                # class-indicating map becomes more faint for low attention
                # regions and the alpha becomes more transparent.  Since those
                # two factors of faintness would otherwise multiplicatively
                # compound we take the square root of both, counteracting the
                # effect.
                gradcam_im = plt.get_cmap("coolwarm")(
                    np.sign(cat_gradcam)
                    * np.sqrt(abs(cat_gradcam) / abs(cat_gradcam).max())
                    / 2
                    + 0.5
                )
                gradcam_im[:, :, -1] = np.sqrt(
                    abs(cat_gradcam) / abs(cat_gradcam).max()
                )
                Image.fromarray(
                    np.uint8(255 * gradcam_im),
                    "RGBA",
                ).resize(
                    np.array(mask.shape)[::-1] * 8, resample=Image.Resampling.NEAREST
                ).save(
                    outdir / f"gradcam_{target_label}_{category}.png",
                    pnginfo=make_metadata(
                        version="barspoon-cat-gradcam 1.0",
                        filename=h5_path.stem,
                        stride=str(stride / 8),
                        target_label=target_label,
                        category=category,
                        scale=f"{abs(cat_gradcam).max():e}",
                    ),
                )


def compute_gradcams(
    *,
    model: LitEncDecTransformer,
    feats: torch.Tensor,
    coords: torch.Tensor,
    stride: int,
    categories: Mapping[str, Sequence[str]],
    batch_size: int,
) -> npt.NDArray[np.float_]:
    """Computes a stack of attention maps

    Returns:
        An array of shape [n_dim, x, y]
    """
    n_outputs = sum(len(c) for c in categories.values())
    flattened_categories = [
        (target, cat_idx, category)
        for target, cs in categories.items()
        for cat_idx, category in enumerate(cs)
    ]

    # We need one gradcam per output class.  We thus replicate the feature
    # tensor `n_outputs` times so we can obtain a gradient map for each of them.
    # Since this would be infeasible for models with many targets, we don't
    # calculate all gradcams at once, but do it in batches instead.
    gradcams = []
    feats_t = feats.unsqueeze(0).repeat(min(batch_size, n_outputs), 1, 1)
    # `feats_t` now has the shape [n_outs, n_tiles, n_features]
    # or [batch_size, n_tiles, n_features], whichever is smaller
    model = model.eval()
    for idx in range(0, n_outputs, batch_size):
        feats_t = feats_t.detach()  # Zero grads of input features
        feats_t.requires_grad = True
        model.zero_grad()
        scores = model.predict_step((feats_t, coords.type_as(feats_t)), batch_idx=0)

        # TODO update this comment; the sentiment is all true
        # Now we have a stack of predictions for each class.  All the rows
        # should be exactly the same, as they only depend on the (repeated and
        # thus identical) tile features.  If we now take the diagonal values
        # `y_i`, the output of the n-th entry _exclusively_ depends on the n-th
        # feature repetition.  If we now calculate `(d sum y_i)/dx`, the n-th
        # repetition of tile features' tensor's gradient will contain exactly
        # `dy_i/dx`.
        sum(
            scores[target_label][i, cat_idx]
            for i, (target_label, cat_idx, _) in enumerate(
                flattened_categories[idx : idx + batch_size]
            )
        ).backward()

        gradcam = (feats_t.grad * feats_t).sum(-1)
        gradcams.append(gradcam.detach().cpu())

    # If n_outs isn't divisible by batch_size, we'll have some superfluous
    # output maps which we have to drop
    gradcams = torch.cat(gradcams)[:n_outputs]
    gradcams_2d = vals_to_im(
        gradcams.permute(1, 0).float().numpy(),
        coords.to(torch.int).cpu().numpy(),
        stride,
    )

    unflattened_gradcams_2d = defaultdict(dict)
    for (target_label, _, category), gradcam_2d in zip(
        flattened_categories, gradcams_2d
    ):
        unflattened_gradcams_2d[target_label][category] = gradcam_2d
    return unflattened_gradcams_2d


def vals_to_im(
    scores: npt.NDArray[Any], coords: npt.NDArray[np.int_], stride: int
) -> npt.NDArray[np.float_]:
    """Converts linear arrays of scores, coords into a 2d array

    Args:
        scores: An array of scores [n_tiles, ...]
        coords: An array of coordinates [n_tiles, ...], relating each tile to a
            position
    """
    size = coords.max(0)[::-1] // stride + 1
    im = np.zeros((scores.shape[-1], *size))
    for s, c in zip(scores, coords, strict=True):
        im[:, c[1] // stride, c[0] // stride] = s
    return im


def make_metadata(**kwargs) -> PngInfo:
    """Helper function to generate a PNG metadata dict"""
    metadata = PngInfo()
    for k, v in kwargs.items():
        metadata.add_text(k, v)
    return metadata


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

    parser.add_argument("--batch-size", type=int, default=0x20)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    return parser


if __name__ == "__main__":
    main()
