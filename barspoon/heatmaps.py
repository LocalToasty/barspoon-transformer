import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from tqdm import tqdm

from .model import LitEncDecTransformer
from .utils import make_dataset_df


def main():
    parser = make_argument_parser()
    args = parser.parse_args()

    pl.seed_everything(0)
    torch.set_float32_matmul_precision("medium")

    model = LitEncDecTransformer.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path
    ).to(torch.bfloat16)

    target_labels = model.hparams["target_labels"]

    # load dataset, grouped by filename
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
        ), "we have grouped by filename, so there should only be one slide per item"
        h5_path = slides[0]

        # load features
        with h5py.File(h5_path) as h5_file:
            feats = h5_file["feats"][:]
            coords = h5_file["coords"][:]
            xs = np.sort(np.unique(coords[:, 0]))
            stride = np.min(xs[1:] - xs[:-1])

        att_maps = compute_attention_maps(model, feats, coords, stride, args.batch_size)
        score_maps = compute_score_maps(model, feats, coords, stride)

        # save all the heatmaps
        (outdir := args.output_dir / h5_path.stem).mkdir(exist_ok=True, parents=True)
        for i, target_label in enumerate(target_labels):
            # save metadata in slide
            metadata = PngInfo()
            metadata.add_text("filename", h5_path.stem)
            metadata.add_text("stride", str(stride))
            metadata.add_text("target_label", target_label)

            Image.fromarray(
                np.uint8(255 * plt.get_cmap("magma")(att_maps[i] / att_maps[i].max())),
                "RGBA",
            ).save(outdir / f"attention_{target_label}.png", pnginfo=metadata)

            Image.fromarray(
                np.uint8(255 * plt.get_cmap("coolwarm")(score_maps[i])), "RGBA"
            ).save(outdir / f"score_{target_label}.png", pnginfo=metadata)

            # use scores as color, attention as alpha
            im = plt.get_cmap("coolwarm")(score_maps[i])
            im[:, :, -1] = att_maps[i] / att_maps[i].max()

            Image.fromarray(np.uint8(255 * im), "RGBA").save(
                outdir / f"map_{target_label}.png", pnginfo=metadata
            )


def compute_attention_maps(
    model, feats, coords, stride, batch_size
) -> npt.NDArray[np.float_]:
    """Computes a stack of attention maps

    Returns:
        An array of shape [n_dim, x, y]
    """
    n_targs = len(model.hparams["target_labels"])
    atts = []
    feats_t = (
        torch.tensor(feats)
        .unsqueeze(0)
        .to(torch.bfloat16)
        .repeat(batch_size, 1, 1)
        .cuda()
    )
    for idx in range(0, n_targs, batch_size):
        feats_t = feats_t.detach()  # zero grads of input features
        feats_t.requires_grad = True
        model.zero_grad()
        scores = model(feats_t).sigmoid()
        scores[:, idx:].diag().sum().backward()

        gradcam = (feats_t.grad * feats_t).abs().sum(-1)
        atts.append(gradcam.detach().cpu())

    atts = torch.cat(atts)[:n_targs]
    return vals_to_im(atts.permute(1, 0), coords, stride)


def compute_score_maps(model, feats, coords, stride):
    """Computes a stack of attention maps

    Returns:
        An array of shape [n_dim, x, y]
    """
    with torch.no_grad():
        feats_t = torch.tensor(feats).to(torch.bfloat16).unsqueeze(1).cuda()
        scores = model(feats_t).sigmoid().detach().cpu()
    score_maps = vals_to_im(scores, coords, stride)
    return score_maps


def vals_to_im(scores, coords, stride) -> npt.NDArray[np.float_]:
    """Converts linear arrays of scores, coords into a 2d array"""
    size = coords.max(0)[::-1] // stride + 1
    im = np.zeros((scores.shape[-1], *size))
    for s, c in zip(scores, coords, strict=True):
        im[:, c[1] // stride, c[0] // stride] = s.float()
    return im


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
        action="append",
        help="Path to the clinical table. Can be specified multiple times",
    )
    parser.add_argument(
        "-s",
        "--slide-table",
        metavar="PATH",
        dest="slide_tables",
        type=Path,
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

    parser.add_argument(
        "-m",
        "--checkpoint-path",
        metavar="PATH",
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
