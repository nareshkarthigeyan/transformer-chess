# src/model.py
import torch
import torch.nn as nn

class ChessTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_feedforward=512):
        super(ChessTransformer, self).__init__()
        
        # 13 piece tokens (0=empty, 1-6=White, 7-12=Black)
        self.piece_embedding = nn.Embedding(13, d_model)
        
        # PILLAR 1: 2D Spatial Grid Embeddings
        self.row_embedding = nn.Embedding(8, d_model)
        self.col_embedding = nn.Embedding(8, d_model)
        
        # Build coordinate static lookup tensors for quick mapping
        self.register_buffer("row_indices", torch.arange(8).repeat_interleave(8))
        self.register_buffer("col_indices", torch.arange(8).repeat(8))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            activation='gelu', 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # AlphaZero-style heads: policy over moves plus a scalar board value.
        flattened_dim = 64 * d_model
        self.fc_out = nn.Linear(flattened_dim, 4096)
        self.value_head = nn.Sequential(
            nn.Linear(flattened_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Tanh(),
        )

    def _embed_board(self, x):
        """Build the spatially-aware token stream used by the encoder."""
        batch_size = x.size(0)
        
        # 1. Fetch categorical piece representations
        x_emb = self.piece_embedding(x)  # Shape: (Batch, 64, d_model)
        
        # 2. Extract and fuse 2D geometric vector signals
        rows = self.row_embedding(self.row_indices).unsqueeze(0).expand(batch_size, -1, -1)
        cols = self.col_embedding(self.col_indices).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Combine everything together
        x = x_emb + rows + cols  # Shape: (Batch, 64, d_model)
        
        return x

    def encode_with_intermediates(self, x):
        """Return the final encoder stream and each layer's residual stream.

        The intermediate streams are intentionally exposed for the web
        dashboard's logit lens. Each one has the same shape as the final
        stream, so the policy head can be applied to it without changing the
        trained model.
        """
        x = self._embed_board(x)
        activations = []
        for layer in self.transformer_encoder.layers:
            x = layer(x)
            activations.append(x)

        if self.transformer_encoder.norm is not None:
            x = self.transformer_encoder.norm(x)

        return x, activations

    def forward(self, x, return_value=False, return_intermediates=False):
        batch_size = x.size(0)
        # 3. Process spatial context correlations through self-attention
        if return_intermediates:
            enc_out, activations = self.encode_with_intermediates(x)
        else:
            enc_out = self.transformer_encoder(self._embed_board(x))
            activations = None
        enc_out = enc_out.contiguous().view(batch_size, -1)

        # 4. Return raw policy logits, and optionally the current-player value.
        policy_logits = self.fc_out(enc_out)
        if return_value and return_intermediates:
            return policy_logits, self.value_head(enc_out).squeeze(-1), activations
        if return_value:
            return policy_logits, self.value_head(enc_out).squeeze(-1)
        if return_intermediates:
            return policy_logits, activations
        return policy_logits
