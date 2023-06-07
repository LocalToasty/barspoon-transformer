import math
import re
from collections.abc import Sequence
from typing import Any, Literal, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import Tensor, nn
from torchmetrics.functional.classification.auroc import _multilabel_auroc_compute
from torchmetrics.utilities.data import dim_zero_cat, select_topk


class EncDecTransformer(nn.Module):
    """An encoder decoder architecture for multilabel classification tasks

    This architecture is a modified encoder decoder stack: First, we encode the
    input tokens using an encoder stack.  We then decode these tokens using a
    set of class tokens, one per output label.  Finally, we forward each of the
    decoded tokens through a fully connected layer to get a label-wise
    prediction.

             +---+
        t -->| E |--+
             +---+  |
                    v
                  +---+   +---+
        c ------->| D |-->|FCs|--> s
                  +---+   +---+
    """

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


class ParallelLinear(nn.Module):
    """Parallelly applies multiple linear layers.

    For an input of shape (N, F) or (B, N, F), this layer applies a separate
    linear layer to each of the N channels of the input.
    """

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
                sanatize(target_label): torchmetrics.classification.BinaryAUROC()
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
        if isinstance(batch, Tensor):
            bag = batch
        else:
            bag, _ = batch
        logits = self(bag)
        return torch.sigmoid(logits)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def sanatize(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", x)


class TopKMultilabelAUROC(torchmetrics.classification.MultilabelPrecisionRecallCurve):
    """Computes the AUROC of the K best-performing targets"""

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

    def compute(self) -> Tensor:  # type: ignore
        state = (dim_zero_cat(self.preds), dim_zero_cat(self.target)) # type: ignore
        individual_aurocs = _multilabel_auroc_compute(
            state, self.num_labels, average="none", thresholds=None
        )
        assert isinstance(individual_aurocs, Tensor)
        topk_idx = select_topk(individual_aurocs, self.topk, dim=0).bool()

        state = (
                dim_zero_cat(self.preds)[:, topk_idx],  # type: ignore
                dim_zero_cat(self.target)[:, topk_idx], # type: ignore
            )
        auroc = _multilabel_auroc_compute(
            state, self.topk, average=self.average, thresholds=None
        )
        assert isinstance(auroc, Tensor)
        return auroc


class LitEncDecTransformer(LitMilClassificationMixin):
    def __init__(
        self,
        *,
        d_features: int,
        target_labels: Sequence[str],
        pos_weight: Optional[torch.Tensor],
        # model parameters
        d_model: int = 512,
        num_encoder_heads: int = 8,
        num_decoder_heads: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 2048,
        # other hparams
        learning_rate: float = 1e-4,
        **hparams: Any,
    ) -> None:
        super().__init__(
            target_labels=target_labels,
            pos_weight=pos_weight,
            learning_rate=learning_rate,
        )
        _ = hparams  # so we don't get unused parameter warnings

        self.model = EncDecTransformer(
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
