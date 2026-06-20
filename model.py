import torch
import torch.nn as nn

class ChessTransformer(nn.Module):
    def __init__(self, vocab_size=13, d_model=128, nhead=4, num_layers=4, num_classes=4096):
        super(ChessTransformer, self).__init__()
        
        # 1. Embedding Layer: Map piece tokens (0-12) to continuous vectors
        self.piece_embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Positional Encoding: Since chess layout matters (A1 is different from H8), 
        # we give each of the 64 squares its own learned position embedding.
        self.pos_embedding = nn.Embedding(64, d_model)
        
        # 3. Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=0.1,
            activation='gelu',
            batch_first=True  # Expected shape: (batch_size, seq_len, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Output Head: Maps the 64 standard features down to move probabilities
        self.fc_out = nn.Sequential(
            nn.Linear(64 * d_model, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes) # Out: 4096 possible move classes
        )
        
    def forward(self, x):
        # x shape: (batch_size, 64)
        batch_size, seq_len = x.shape
        
        # Generate position IDs [0, 1, ..., 63] for the batch
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Combine token embeddings and spatial positions
        x = self.piece_embedding(x) + self.pos_embedding(positions)
        
        # Pass through the transformer layers
        x = self.transformer_encoder(x) # Shape: (batch_size, 64, d_model)
        
        # Flatten the spatial sequence to feed into the dense layer
        x = x.view(batch_size, -1) # Shape: (batch_size, 64 * d_model)
        
        # Predict logits for all possible moves
        logits = self.fc_out(x) # Shape: (batch_size, 4096)
        
        return logits

# --- Quick Test ---
if __name__ == "__main__":
    model = ChessTransformer()
    # Let's mock a batch of 2 chess games using your processed shape
    mock_batch = torch.randint(0, 13, (2, 64)) 
    output = model(mock_batch)
    print("Model Output Shape:", output.shape) # Should be torch.Size([2, 4096])