"""Implementation of relative position representations for multihead attention [1].

Code is adapted from https://github.com/georg-wolflein/das-mil/blob/master/mil/models/distance_aware_self_attention.py.

[1]: https://arxiv.org/abs/2305.10552 "Deep Multiple Instance Learning with Distance-Aware Self-Attention"
"""

from torch import nn
from typing import Optional, Tuple, List, Union, Callable
from torch.nn import functional as F

from torch import nn
import torch.nn.functional as F
import torch


class EmbeddingTable(nn.Module):
    """Embedding table with trainable embeddings."""

    def __init__(self, dim, num_embeddings=2, trainable: bool = True):
        super().__init__()
        self.embeddings = torch.empty(num_embeddings, dim)
        if trainable:
            self.embeddings = nn.Parameter(self.embeddings)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.normal_(self.embeddings, mean=0, std=1)

    def forward(self, weights):
        # weights shape: Nxnum_embeddings
        embeddings = self.embeddings  # num_embeddings x dim
        # embeddings = embeddings * torch.tensor([0, 1]).unsqueeze(-1)
        X = weights @ embeddings  # Nxdim
        return X


class ContinuousEmbeddingIndex(nn.Module):
    """Provides an embedding index for continuous values using interpolation via a sigmoid."""

    def __init__(self, num_embeddings=2):
        super().__init__()
        assert num_embeddings == 2
        self.bias = nn.Parameter(torch.empty(1))
        self.multiplier = nn.Parameter(torch.empty(1))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.bias.data.fill_(.5)
        self.multiplier.data.fill_(10)  # TODO: test different initializations

    def forward(self, x):
        x = torch.sigmoid((x - self.bias) * self.multiplier)
        x = torch.stack([x, 1 - x], dim=-1)
        return x


class DiscreteEmbeddingIndex(nn.Module):
    """Provides an embedding index for discrete values using one-hot encoding."""

    def __init__(self, num_embeddings=2):
        super().__init__()
        self.num_embeddings = num_embeddings

    def forward(self, x):
        # x shape: num_edges x 1
        x = x * (self.num_embeddings - 1)  # assume 0 <= x <= 1
        x = torch.round(x)
        x = torch.clamp(x, 0, self.num_embeddings - 1)
        return F.one_hot(x.long().squeeze(-1), self.num_embeddings).float()


class DistanceAwareMultiheadAttention(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,

        # Custom arguments for relative positional encoding
        embed_keys: bool = True,
        embed_queries: bool = True,
        embed_values: bool = True,
        trainable_embeddings: bool = True,
        continuous: bool = True,
        num_embeddings: int = 2,  # if continuous, must be 2
    ):
        super().__init__()

        # Unsupported arguments
        assert batch_first is True
        assert add_bias_kv is False
        assert add_zero_attn is False
        assert device is None
        assert dtype is None

        kdim = kdim if kdim is not None else embed_dim
        vdim = vdim if vdim is not None else embed_dim

        head_dim = embed_dim // num_heads
        assert (
            head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.q_proj = nn.Linear(embed_dim, kdim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, kdim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, vdim, bias=bias)

        self.dropout = nn.Dropout(dropout)

        if embed_keys:
            self.embed_k = EmbeddingTable(
                kdim // num_heads, num_embeddings=num_embeddings, trainable=trainable_embeddings)
        if embed_queries:
            self.embed_q = EmbeddingTable(
                kdim // num_heads, num_embeddings=num_embeddings, trainable=trainable_embeddings)
        if embed_values:
            self.embed_v = EmbeddingTable(
                vdim // num_heads, num_embeddings=num_embeddings, trainable=trainable_embeddings)

        self.embed_keys = embed_keys
        self.embed_queries = embed_queries
        self.embed_values = embed_values

        EmbeddingIndex = ContinuousEmbeddingIndex if continuous else DiscreteEmbeddingIndex
        self.index_k = EmbeddingIndex(num_embeddings=num_embeddings)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)

    @staticmethod
    def compute_relative_distances(tile_positions: torch.Tensor, max_dist: float = 100_000 * 2**.5):
        """
        Compute pairwise Euclidean distances between all pairs of positions in a tile.
        :param tile_positions: [Batch, SeqLen, 2] tensor of 2D positions
        :param max_dist: maximum distance to normalize by
        :return: [Batch, SeqLen, SeqLen] tensor of distances
        """

        # Compute pairwise differences
        diff = tile_positions.unsqueeze(2) - tile_positions.unsqueeze(1)
        # Compute pairwise distances
        dist = torch.norm(diff, dim=-1)
        if max_dist:
            dist /= max_dist
        return dist

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        tile_positions: torch.Tensor,
        need_weights: bool = False,
        **kwargs
    ):
        batch_size, seq_length, _ = query.shape
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_length, self.num_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        k = k.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        v = v.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]

        #
        # Scaled dot product attention
        #
        d_k = q.size()[-1]
        A = torch.matmul(q, k.transpose(-2, -1))
        # [Batch, Head, SeqLen, SeqLen]

        # Compute additional distance-aware terms for keys/queries
        rel_dists = self.compute_relative_distances(
            tile_positions)  # [Batch, SeqLen, SeqLen]

        # Term 1
        if self.embed_keys:
            rk = self.embed_k(self.index_k(rel_dists)).unsqueeze(-4)
            # rk is shape [Batch, 1, SeqLen, SeqLen, Dims]
            q_repeat = q.unsqueeze(-2)
            # q_repeat is shape [Batch, Head, SeqLen, 1, Dims]
            # A = A + (q_repeat * rk).sum(axis=-1)  # NxN
            A = A + torch.einsum('bhqrd,bhqrd->bhqr', q_repeat, rk)

        # Term 2
        if self.embed_queries:
            rq = self.embed_q(self.index_k(rel_dists)).unsqueeze(-4)
            # rq is shape [Batch, 1, SeqLen, SeqLen, Dims]
            k_repeat = k.unsqueeze(-3)
            # k_repeat is shape [Batch, Head, 1, SeqLen, Dims]
            # A = A + (k_repeat * rq).sum(axis=-1)  # NxN
            A = A + torch.einsum('bhqrd,bhqrd->bhqr', k_repeat, rq)

        # Term 3
        if self.embed_keys and self.embed_queries:
            # A = A + (q_repeat * k_repeat).sum(axis=-1)  # NxN
            A = A + torch.einsum('bhqrd,bhqrd->bhqr', q_repeat, k_repeat)

        # Scale by sqrt(d_k)
        A = A / d_k ** .5
        A = F.softmax(A, dim=-1)

        A_dropout = self.dropout(A)
        values = torch.matmul(A_dropout, v)

        # Unify heads
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, -1)

        if need_weights:
            return values, A_dropout
        return values,


class DistanceAwareTransformerEncoderLayer(nn.Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
        continuous: bool = True,
        bins: int = 2
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = DistanceAwareMultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first,
            continuous=continuous, num_embeddings=bins, **factory_kwargs
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.relu

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), *args, **kwargs)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, *args, **kwargs))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = self.self_attn(x, x, x, *args, **kwargs)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class DistanceAwareTransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super().__init__()
        self.layers = nn.modules.transformer._get_clones(
            encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check

    def forward(
            self,
            src: torch.Tensor,
            tile_positions: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        output = src
        for mod in self.layers:
            output = mod(
                output, *args, tile_positions=tile_positions, **kwargs)

        if self.norm is not None:
            output = self.norm(output)

        return output
