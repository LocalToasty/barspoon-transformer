import argparse
from pathlib import Path

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
        ), "there should only be one slide per item after grouping by slidename"
        h5_path = slides[0]

        # Load features
        with h5py.File(h5_path) as h5_file:
            feats = h5_file["feats"][:]
            coords = h5_file["coords"][:]
            xs = np.sort(np.unique(coords[:, 0]))
            stride = np.min(xs[1:] - xs[:-1])

        att_maps = compute_attention_maps(model, feats, coords, stride, args.batch_size)
        score_maps = compute_score_maps(model, feats, coords, stride)

        mask = (att_maps > 0).any(axis=0)

        categories = model.hparams["categories"]
        target_edges = np.cumsum([0, *(len(c) for c in categories)])

        # Save all the heatmaps
        (outdir := args.output_dir / h5_path.stem).mkdir(exist_ok=True, parents=True)
        for target_label, cs, left in zip(
            target_labels,
            categories,
            target_edges[:-1],
            strict=True,
        ):
            for i, category in enumerate(cs):
                # Save metadata in slide
                att_im = plt.get_cmap("magma")(att_maps[left + i] / att_maps.max())
                att_im[:, :, -1] = mask
                Image.fromarray(
                    np.uint8(255 * att_im),
                    "RGBA",
                ).save(
                    outdir / f"attention_{target_label}_{category}.png",
                    pnginfo=make_metadata(
                        # metadata format semver
                        # update minor version when adding fields,
                        # major when removing fields / changing semantics
                        version="barspoon-attention 1.0",
                        filename=h5_path.stem,
                        stride=str(stride),
                        target_label=target_label,
                        category=category,
                        attention_scale=f"{att_maps.max():e}",
                    ),
                )

                # use scores as color, attention as alpha
                im = plt.get_cmap("coolwarm")(score_maps[left + i])
                im[:, :, -1] = att_maps[left + i] / att_maps[left + i].max()

                Image.fromarray(np.uint8(255 * im), "RGBA").save(
                    outdir / f"map_{target_label}_{category}.png",
                    pnginfo=make_metadata(
                        version="barspoon-map 1.0",  # see above
                        filename=h5_path.stem,
                        stride=str(stride),
                        target_label=target_label,
                        category=category,
                        attention_scale=f"{att_maps[left+i].max():e}",
                    ),
                )


def compute_attention_maps(
    model, feats, coords, stride, batch_size
) -> npt.NDArray[np.float_]:
    """Computes a stack of attention maps

    Returns:
        An array of shape [n_dim, x, y]
    """
    n_outs = sum(len(w) for w in model.weights)
    atts = []
    feats_t = (
        torch.tensor(feats)
        .unsqueeze(0)
        .to(torch.bfloat16)
        .repeat(batch_size, 1, 1)
        .cuda()
    )
    model = model.eval()
    for idx in range(0, n_outs, batch_size):
        feats_t = feats_t.detach()  # Zero grads of input features
        feats_t.requires_grad = True
        model.zero_grad()
        scores = model(feats_t).sigmoid()
        scores[:, idx:].diag().sum().backward()

        gradcam = (feats_t.grad * feats_t).abs().sum(-1)
        atts.append(gradcam.detach().cpu())

    # If n_outs isn't divisible by batch_size,
    # we'll have some superfluous output maps
    # Drop them
    atts = torch.cat(atts)[:n_outs]
    return vals_to_im(atts.permute(1, 0), coords, stride)


def compute_score_maps(model, feats, coords, stride):
    """Computes a stack of attention maps

    Returns:
        An array of shape [n_dim, x, y]
    """
    model = model.eval()
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


def make_metadata(**kwargs) -> PngInfo:
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
