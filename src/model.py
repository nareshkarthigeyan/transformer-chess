import torch
import torch.nn as nn

class ChessTransformer(nn.Module):
    def __init__(self, vocab_size=13, d_model=128, nhead=4, num_layers=4, num_classes=4096):
        super(ChessTransformer, self).__init__()
        
        self.piece_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(64, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Sequential(
            nn.Linear(64 * d_model, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        x = self.piece_embedding(x) + self.pos_embedding(positions)
        x = self.transformer_encoder(x)
        x = x.view(batch_size, -1)
        logits = self.fc_out(x)
        
        return logits