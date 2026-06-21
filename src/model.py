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
        
        # Output classification head maps to 4,096 explicit move combinations
        self.fc_out = nn.Linear(64 * d_model, 4096)

    def forward(self, x):
        batch_size = x.size(0)
        
        # 1. Fetch categorical piece representations
        x_emb = self.piece_embedding(x)  # Shape: (Batch, 64, d_model)
        
        # 2. Extract and fuse 2D geometric vector signals
        rows = self.row_embedding(self.row_indices).unsqueeze(0).expand(batch_size, -1, -1)
        cols = self.col_embedding(self.col_indices).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Combine everything together
        x = x_emb + rows + cols  # Shape: (Batch, 64, d_model)
        
        # 3. Process spatial context correlations through self-attention
        enc_out = self.transformer_encoder(x)
        enc_out = enc_out.contiguous().view(batch_size, -1)
        
        # 4. Return raw logits
        return self.fc_out(enc_out)