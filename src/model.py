"""Geometry-aware transformer used by the chess policy/value model.

The board is represented as 64 square tokens. Unlike a plain transformer,
every attention layer receives an additive relative-geometry bias based on the
Manhattan distance between two squares. Nearby squares are initially favoured
and the bias is learned during training.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryBiasedEncoderLayer(nn.Module):
    """Pre-norm transformer block with a learned square-distance bias."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, geometry_bias: torch.Tensor) -> torch.Tensor:
        residual = x
        normalized = self.norm1(x)
        attended, _ = self.self_attn(
            normalized, normalized, normalized, attn_mask=geometry_bias, need_weights=False
        )
        x = residual + self.dropout1(attended)
        residual = x
        x = self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(x)))))
        return residual + self.dropout2(x)


class GeometryBiasedTransformerEncoder(nn.Module):
    """A small encoder which can also expose residual streams for Logic Lens."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GeometryBiasedEncoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, geometry_bias: torch.Tensor, return_intermediates: bool = False
    ):
        activations = []
        for layer in self.layers:
            x = layer(x, geometry_bias)
            if return_intermediates:
                activations.append(x)
        x = self.norm(x)
        return (x, activations) if return_intermediates else x


class ChessTransformer(nn.Module):
    """Board-to-policy/value transformer with explicit geometric attention.

    ``state_features`` is optional so legacy callers that only pass the 64
    board tokens still work. New datasets provide side to move, castling,
    en-passant, halfmove, and fullmove buckets. They are embedded into a global
    board context and added to every square token.
    """

    state_vocab_sizes: Tuple[int, ...] = (2, 16, 65, 101, 256)

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")

        self.piece_embedding = nn.Embedding(13, d_model)
        self.row_embedding = nn.Embedding(8, d_model)
        self.col_embedding = nn.Embedding(8, d_model)
        self.state_embeddings = nn.ModuleList(
            nn.Embedding(size, d_model) for size in self.state_vocab_sizes
        )
        # Keeps old board-only checkpoints behaviourally stable on first load;
        # fresh distillation learns the global-state contribution immediately.
        for embedding in self.state_embeddings:
            nn.init.zeros_(embedding.weight)

        self.register_buffer("row_indices", torch.arange(8).repeat_interleave(8))
        self.register_buffer("col_indices", torch.arange(8).repeat(8))
        manhattan = (
            (self.row_indices[:, None] - self.row_indices[None, :]).abs()
            + (self.col_indices[:, None] - self.col_indices[None, :]).abs()
        )
        self.register_buffer("geometry_distance", manhattan.long())
        # Distance 0..14. The mild negative initialization makes locality an
        # actual inductive bias from step zero, while keeping it fully learnable.
        self.geometry_bias_by_distance = nn.Parameter(-0.03 * torch.arange(15.0))

        self.transformer_encoder = GeometryBiasedTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        flattened_dim = 64 * d_model
        self.fc_out = nn.Linear(flattened_dim, 4096)
        self.value_head = nn.Sequential(
            nn.Linear(flattened_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Tanh(),
        )
        self._config = {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
        }

    @property
    def config(self) -> Dict[str, int | float]:
        return dict(self._config)

    @classmethod
    def from_config(cls, config: Optional[Dict[str, int | float]]) -> "ChessTransformer":
        return cls(**(config or {}))

    def geometry_bias(self) -> torch.Tensor:
        """Return the additive [64, 64] attention bias used by every layer."""
        return self.geometry_bias_by_distance[self.geometry_distance]

    def geometry_profile(self) -> Iterable[float]:
        """Presentation-safe serialisable distance-to-attention-bias profile."""
        return self.geometry_bias_by_distance.detach().cpu().tolist()

    def _embed_board(
        self, x: torch.Tensor, state_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x.ndim != 2 or x.size(1) != 64:
            raise ValueError("Expected board tokens with shape [batch, 64].")
        batch_size = x.size(0)
        embedded = self.piece_embedding(x)
        rows = self.row_embedding(self.row_indices).unsqueeze(0).expand(batch_size, -1, -1)
        cols = self.col_embedding(self.col_indices).unsqueeze(0).expand(batch_size, -1, -1)
        embedded = embedded + rows + cols

        if state_features is not None:
            if state_features.ndim != 2 or state_features.size(1) != len(self.state_vocab_sizes):
                raise ValueError(
                    f"Expected state features with shape [batch, {len(self.state_vocab_sizes)}]."
                )
            state_context = torch.zeros(
                batch_size, embedded.size(-1), device=embedded.device, dtype=embedded.dtype
            )
            for index, embedding in enumerate(self.state_embeddings):
                values = state_features[:, index].long().clamp(0, embedding.num_embeddings - 1)
                state_context = state_context + embedding(values)
            embedded = embedded + state_context.unsqueeze(1)
        return embedded

    def encode_with_intermediates(
        self, x: torch.Tensor, state_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        embedded = self._embed_board(x, state_features)
        return self.transformer_encoder(embedded, self.geometry_bias(), return_intermediates=True)

    def forward(
        self,
        x: torch.Tensor,
        state_features: Optional[torch.Tensor] = None,
        return_value: bool = False,
        return_intermediates: bool = False,
    ):
        batch_size = x.size(0)
        if return_intermediates:
            encoded, activations = self.encode_with_intermediates(x, state_features)
        else:
            encoded = self.transformer_encoder(
                self._embed_board(x, state_features), self.geometry_bias()
            )
            activations = None
        flattened = encoded.contiguous().view(batch_size, -1)
        policy_logits = self.fc_out(flattened)
        if return_value and return_intermediates:
            return policy_logits, self.value_head(flattened).squeeze(-1), activations
        if return_value:
            return policy_logits, self.value_head(flattened).squeeze(-1)
        if return_intermediates:
            return policy_logits, activations
        return policy_logits
