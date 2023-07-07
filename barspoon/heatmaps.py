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
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from packaging.specifiers import Specifier
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from tqdm import tqdm

from barspoon.model import LitEncDecTransformer
from barspoon.utils import make_dataset_df


def main():
    parser = make_argument_parser()
    args = parser.parse_args()

    # Load model and ensure its version is compatible
    model = LitEncDecTransformer.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path
    ).to(torch.bfloat16)
    name, version = model.hparams["version"].split(" ")
    if not (
        name == "barspoon-transformer"
        and (spec := Specifier("~=1.0")).contains(version)
    ):
        raise ValueError(
            f"model not compatible with barspoon-transformer {spec}",
            model.hparams["version"],
        )

    target_labels = model.hparams["target_labels"]

    # Load dataset, grouped by filename
    dataset_df = make_dataset_df(
        clini_tables=args.clini_tables,
        slide_tables=args.slide_tables,
        feature_dirs=args.feature_dirs,
        patient_col=args.patient_col,
        filename_col=args.filename_col,
        group_by=args.filename_col,
        target_labels=target_labels,
    )

    for slides in tqdm(dataset_df.path):
        assert (
            len(slides) == 1
        ), "there should only be one slide per item after grouping by slidename"
        h5_path = slides[0]

        # Load features
        with h5py.File(h5_path) as h5_file:
            feats = h5_file["feats"][:]
            coords = h5_file["coords"][:]
            xs = np.sort(np.unique(coords[:, 0]))
            stride = np.min(xs[1:] - xs[:-1])

        # Generate the gradcams
        # If something goes wrong (most probably an OOM error),
        # just skip the slide
        try:
            gradcams = compute_attention_maps(
                model, feats, coords, stride, args.batch_size
            )
        except Exception as exception:
            logging.error(f"error while processing {slides[0]}: {exception})")
            continue

        mask = (gradcams > 0).any(axis=0)

        categories = model.hparams["categories"]
        target_edges = np.cumsum([0, *(len(c) for c in categories)])

        # Save all the heatmaps
        (outdir := args.output_dir / h5_path.stem).mkdir(exist_ok=True, parents=True)
        for target_label, cs, left, right in zip(
            target_labels,
            categories,
            # The indices at which the scores belonging to the target
            # start / end
            target_edges[:-1],
            target_edges[1:],
            strict=True,
        ):
            gradcam_im = plt.get_cmap("magma")(
                abs(gradcams[left:right]).mean(0) / abs(gradcams).max()
            )
            gradcam_im[:, :, -1] = mask
            Image.fromarray(
                np.uint8(255 * gradcam_im),
                "RGBA",
            ).resize(
                np.array(mask.shape)[::-1] * 8, resample=Image.Resampling.NEAREST
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
                    scale=f"{abs(gradcams).max():e}",
                ),
            )
            for category, cat_gradcam in zip(cs, gradcams[left:right], strict=True):
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


def compute_attention_maps(
    model, feats, coords, stride, batch_size
) -> npt.NDArray[np.float_]:
    """Computes a stack of attention maps

    Returns:
        An array of shape [n_dim, x, y]
    """
    # We need one gradcam per output class.
    # We thus replicate the feature tensor `n_outputs` times
    # so we can obtain a gradient map for each of them
    # Since this would be infeasible for models with many targets,
    # we don't calculate all gradcams at once, but do it in batches instead
    n_outputs = sum(len(w) for w in model.weights)
    atts = []
    feats_t = (
        torch.tensor(feats)
        .unsqueeze(0)
        .to(torch.bfloat16)
        .repeat(min(batch_size, n_outputs), 1, 1)
        .cuda()  # TODO don't hard-code
    )
    # `feats_t` now has the shape [n_outs, n_tiles, n_features]
    # or [batch_size, n_tiles, n_features], whichever is smaller
    model = model.eval()
    for idx in range(0, n_outputs, batch_size):
        feats_t = feats_t.detach()  # Zero grads of input features
        feats_t.requires_grad = True
        model.zero_grad()
        scores = model.predict_step(feats_t, batch_idx=0)
        # Now we have a stack of predictions for each class.  All the rows
        # should be exactly the same, as they only depend on the (repeated and
        # thus identical) tile features.  If we now take the diagonal values
        # `y_i`, the output of the n-th entry _exclusively_ depends on the n-th
        # feature repetition.  If we now calculate `(d sum y_i)/dx`, the n-th
        # repetition of tile features' tensor's gradient will contain exactly
        # `dy_i/dx`.
        scores[:, idx:].diag().sum().backward()

        gradcam = (feats_t.grad * feats_t).sum(-1)
        atts.append(gradcam.detach().cpu())

    # If n_outs isn't divisible by batch_size, we'll have some superfluous
    # output maps which we have to drop
    atts = torch.cat(atts)[:n_outputs]
    return vals_to_im(atts.permute(1, 0), coords, stride)


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
        im[:, c[1] // stride, c[0] // stride] = s.float()
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
        metavar="SLIDE_TABLE",
        dest="slide_tables",
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

    parser.add_argument("--batch-size", type=int, default=0x20)

    return parser


if __name__ == "__main__":
    main()
