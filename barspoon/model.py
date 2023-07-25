import random
import re
from typing import Any, Dict, Mapping, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import Tensor, nn
from torchmetrics.utilities.data import dim_zero_cat

__all__ = [
    "LitEncDecTransformer",
    "EncDecTransformer",
    "LitMilClassificationMixin",
    "SafeMulticlassAUROC",
]


class EncDecTransformer(nn.Module):
    """An encoder decoder architecture for multilabel classification tasks

    This architecture is a modified version of the one found in [Attention Is
    All You Need][1]: First, we project the features into a lower-dimensional
    feature space, to prevent the transformer architecture's complexity from
    exploding for high-dimensional features.  We add sinusodial [positional
    encodings][1].  We then encode these projected input tokens using a
    transformer encoder stack.  Additional tabular data is also encoded into
    tokens using a stack of to fully connected layers per label.  These encoded
    tokens are then concatenated to the sequene of encoded tile tokens.  Next,
    we decode these tokens using a set of class tokens, one per output label.
    Finally, we forward each of the decoded tokens through a fully connected
    layer to get a label-wise prediction.

                  PE1
                   |
             +--+  v   +---+
        t1 ->|FC|--+-->|   |--+     // Encoded tile tokens
         .   +--+      | E |  |
         .             | x |  |
         .   +--+      | n |  |
        tn ->|FC|--+-->|   |--+
             +--+  ^   +---+  |
                   |          |
                  PEn         |
                              |
             +---+            |
        a1 ->|AE1|------------+     // Encoded tabular data tokens
         .   +---+            |
         .                    |
         .   +---+            |
        am ->|AEm|------------+
             +---+            |
                              v
                            +---+   +---+
        c1 ---------------->|   |-->|FC1|--> s1
         .                  | D |   +---+     .
         .                  | x |             .
         .                  | k |   +---+     .
        ck ---------------->|   |-->|FCk|--> sk
                            +---+   +---+

    We opted for this architecture instead of a more traditional [Vision
    Transformer][2] to improve performance for multi-label predictions with many
    labels.  Our experiments have shown that adding too many class tokens to a
    vision transformer decreases its performance, as the same weights have to
    both process the tiles' information and the class token's processing.  Using
    an encoder-decoder architecture alleviates these issues, as the data-flow of
    the class tokens is completely independent of the encoding of the tiles.
    Furthermore, analysis has shown that there is almost no interaction between
    the different classes in the decoder.  While this points to the decoder
    being more powerful than needed in practice, this also means that each
    label's prediction is mostly independent of the others.  As a consequence,
    noisy labels will not negatively impact the accuracy of non-noisy ones.

    In our experiments so far we did not see any improvement by adding
    positional encodings.  We tried

     1. [Sinusodal encodings][1]
     2. Adding absolute positions to the feature vector, scaled down so the
        maximum value in the training dataset is 1.

    Since neither reduced performance and the author percieves the first one to
    be more elegant (as the magnitude of the positional encodings is bounded),
    we opted to keep the positional encoding regardless in the hopes of it
    improving performance on future tasks.

    The architecture _differs_ from the one descibed in [Attention Is All You
    Need][1] as follows:

     1. There is an initial projection stage to reduce the dimension of the
        feature vectors and allow us to use the transformer with arbitrary
        features.
     2. Instead of the language translation task described in [Attention Is All
        You Need][1], where the tokens of the words translated so far are used
        to predict the next word in the sequence, we use a set of fixed, learned
        class tokens in conjunction with equally as many independent fully
        connected layers to predict multiple labels at once.

    [1]: https://arxiv.org/abs/1706.03762 "Attention Is All You Need"
    [2]: https://arxiv.org/abs/2010.11929
        "An Image is Worth 16x16 Words:
         Transformers for Image Recognition at Scale"
    """

    def __init__(
        self,
        d_features: int,
        target_n_outs: Dict[str, int],
        *,
        d_model: int = 512,
        num_encoder_heads: int = 8,
        num_decoder_heads: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 2048,
        additional_encoders: Optional[Mapping[str, nn.Module]] = None,
        positional_encoding: bool = True,
    ) -> None:
        super().__init__()

        self.projector = nn.Sequential(
            LearnedNorm(d_features, ignore_zeros=True),
            nn.Linear(d_features, d_model),
            nn.ReLU(),
        )

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

        self.target_labels = target_n_outs.keys()

        # One class token per output label
        self.class_tokens = nn.ParameterDict(
            {
                sanitize(target_label): torch.rand(d_model)
                for target_label in target_n_outs
            }
        )

        if additional_encoders:
            self.additional_encoders = nn.ModuleDict(
                {sanitize(k): v for k, v in additional_encoders.items()}
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

        self.heads = nn.ModuleDict(
            {
                sanitize(target_label): nn.Linear(
                    in_features=d_model, out_features=n_out
                )
                for target_label, n_out in target_n_outs.items()
            }
        )

        self.positional_encoding = positional_encoding

    def forward(
        self,
        tile_tokens: torch.Tensor,
        tile_positions: torch.Tensor,
        additional_features,
    ) -> Dict[str, torch.Tensor]:
        batch_size, _, _ = tile_tokens.shape

        # Encoder stage

        tile_tokens = self.projector(tile_tokens)  # shape: [bs, seq_len, d_model]

        if self.positional_encoding:
            # Add positional encodings
            d_model = tile_tokens.size(-1)
            x = tile_positions.unsqueeze(-1) / 100_000 ** (
                torch.arange(d_model // 4).type_as(tile_positions) / d_model
            )
            positional_encodings = torch.cat(
                [
                    torch.sin(x).flatten(start_dim=-2),
                    torch.cos(x).flatten(start_dim=-2),
                ],
                dim=-1,
            )
            tile_tokens = tile_tokens + positional_encodings

        # Grab both the tile tokens as well as the additional tokens and put
        # aggregate them in a single tensor

        tile_tokens = self.transformer_encoder(tile_tokens)

        if additional_features:
            additional_tokens = torch.cat(
                [
                    self.additional_encoders[sanitize(label)](
                        x.type_as(tile_tokens)
                    ).unsqueeze(
                        -2
                    )  # shape: [bs, 1, d_model]
                    for label, x in additional_features.items()
                ],
                dim=-2,
            )  # shape: [bs, len(additional_features), d_model]

            memory = torch.cat([tile_tokens, additional_tokens], dim=-2)
        else:
            memory = tile_tokens

        # Decoder stage

        class_tokens = torch.stack(
            [self.class_tokens[sanitize(t)] for t in self.target_labels]
        ).expand(batch_size, -1, -1)
        class_tokens = self.transformer_decoder(tgt=class_tokens, memory=memory)

        # Apply the corresponding head to each class token
        logits = {
            target_label: self.heads[sanitize(target_label)](class_token)
            for target_label, class_token in zip(
                self.target_labels,
                class_tokens.permute(1, 0, 2),  # Permute to [target, batch, d_model]
                strict=True,
            )
        }

        return logits


class LearnedNorm(nn.Module):
    """Adaptively learns inputs' stats and transforms the inputs into z-scores

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """

    def __init__(self, d_feats, ignore_zeros=False):
        super().__init__()
        self.register_buffer("n", torch.zeros(1))
        self.register_buffer("mean", torch.zeros(d_feats))
        self.register_buffer("m2", torch.zeros(d_feats))
        self.ignore_zeros = ignore_zeros

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            n_a, mean_a, m2_a = self.n, self.mean, self.m2

            x_flat = torch.flatten(x, end_dim=-2) if x.ndim > 1 else x
            if self.ignore_zeros:
                # For tile embeddings we pad slides with too few tiles with
                # all zero feature vectors.  These would mess up our running
                # stats, so we filter them out.
                x_flat = x_flat[(x_flat != 0).any(-1)]

            n_b = x_flat.shape[0]
            mean_b = x_flat.mean(dim=0)
            m2_b = x_flat.var(dim=0) * n_b

            self.n = (n_a + n_b).detach()
            delta = mean_b - mean_a
            self.mean = (mean_a + delta * n_b / self.n).detach()
            self.m2 = (m2_a + m2_b + delta**2 * n_a * n_b / self.n).detach()

        if self.n < 2:
            return x
        else:
            return (x - self.mean) / self.std

    @property
    def std(self) -> torch.Tensor:
        var = self.m2 / (self.n - 1)
        return torch.sqrt(var)


class LitMilClassificationMixin(pl.LightningModule):
    """Makes a module into a multilabel, multiclass Lightning one"""

    def __init__(
        self,
        *,
        weights: Dict[str, torch.Tensor],
        # Other hparams
        learning_rate: float = 1e-4,
        **hparams: Any,
    ) -> None:
        super().__init__()
        _ = hparams  # So we don't get unused parameter warnings

        self.learning_rate = learning_rate

        target_aurocs = torchmetrics.MetricCollection(
            {
                sanitize(target_label): SafeMulticlassAUROC(num_classes=len(weight))
                for target_label, weight in weights.items()
            }
        )
        for step_name in ["train", "val", "test"]:
            setattr(
                self,
                f"{step_name}_target_aurocs",
                target_aurocs.clone(prefix=f"{step_name}_"),
            )

        self.weights = weights

        self.save_hyperparameters()

    def step(self, batch: Tuple[Tensor, Tensor], step_name=None):
        feats, coords, targets, additional_features = batch

        if step_name == "train":
            selected_additional_features = set(
                random.sample(
                    sorted(additional_features),
                    k=int(random.random() * len(additional_features)),
                )
            )
        else:
            selected_additional_features = {}
        logits = self(
            feats,
            coords,
            additional_features={
                k: additional_features[k] for k in selected_additional_features
            },
        )

        # Calculate the cross entropy loss for each target, then sum them
        loss = sum(
            F.cross_entropy(
                (l := logits[target_label]),
                targets[target_label].type_as(l),
                weight=weight.type_as(l),
            )
            for target_label, weight in self.weights.items()
            if target_label not in selected_additional_features
        )

        if step_name:
            self.log(
                f"{step_name}_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

            # Update target-wise metrics
            for target_label in self.weights:
                target_auroc = getattr(self, f"{step_name}_target_aurocs")[
                    sanitize(target_label)
                ]
                is_na = (targets[target_label] == 0).all(dim=1)
                target_auroc.update(
                    logits[target_label][~is_na],
                    targets[target_label][~is_na].argmax(dim=1),
                )
                self.log(
                    f"{step_name}_{target_label}_auroc",
                    target_auroc,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        return loss

    def training_step(self, batch, batch_idx):
        _ = batch_idx  # Unused
        return self.step(batch, step_name="train")

    def validation_step(self, batch, batch_idx):
        _ = batch_idx  # Unused
        return self.step(batch, step_name="val")

    def test_step(self, batch, batch_idx):
        _ = batch_idx  # Unused
        return self.step(batch, step_name="test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        _, _ = batch_idx, dataloader_idx  # Unused
        if len(batch) == 2:
            feats, positions, additional_features = batch
        else:
            feats, positions, _, additional_features = batch
        logits = self(feats, positions, additional_features=additional_features)

        softmaxed = {
            target_label: torch.softmax(x, 1) for target_label, x in logits.items()
        }
        return softmaxed

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def sanitize(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", x)


class SafeMulticlassAUROC(torchmetrics.classification.MulticlassAUROC):
    """A Multiclass AUROC that doesn't blow up when no targets are given"""

    def compute(self) -> torch.Tensor:
        # Add faux entry if there are none so far
        if len(self.preds) == 0:
            self.update(torch.zeros(1, self.num_classes), torch.zeros(1).long())
        elif len(dim_zero_cat(self.preds)) == 0:
            self.update(
                torch.zeros(1, self.num_classes).type_as(self.preds[0]),
                torch.zeros(1).long().type_as(self.target[0]),
            )
        return super().compute()


class LitEncDecTransformer(LitMilClassificationMixin):
    def __init__(
        self,
        *,
        d_features: int,
        weights: Mapping[str, torch.Tensor],
        # Model parameters
        d_model: int = 512,
        num_encoder_heads: int = 8,
        num_decoder_heads: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 2048,
        positional_encoding: bool = True,
        # Additional Modalities
        additional_inputs: Optional[Mapping[str, int]] = None,
        # Other hparams
        learning_rate: float = 1e-4,
        **hparams: Any,
    ) -> None:
        super().__init__(
            weights=weights,
            learning_rate=learning_rate,
        )
        _ = hparams  # So we don't get unused parameter warnings

        self.model = EncDecTransformer(
            d_features=d_features,
            target_n_outs={t: len(w) for t, w in weights.items()},
            d_model=d_model,
            num_encoder_heads=num_encoder_heads,
            num_decoder_heads=num_decoder_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            positional_encoding=positional_encoding,
            additional_encoders={
                label: nn.Sequential(
                    LearnedNorm(n_feats, ignore_zeros=True),
                    nn.Linear(
                        in_features=n_feats, out_features=(n_feats + d_model) // 2
                    ),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(
                        in_features=(n_feats + d_model) // 2, out_features=d_model
                    ),
                )
                for label, n_feats in (additional_inputs or {}).items()
            },
        )

        self.save_hyperparameters()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
