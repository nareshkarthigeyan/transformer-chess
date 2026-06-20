import torch
import torch.nn as nn
from src.data_loader import process_pgn_games
from src.model import ChessTransformer
from src.train import train_epoch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare sample data
    sample_pgn = "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 1-0"
    X_train, Y_train = process_pgn_games(sample_pgn)
    print(f"Processed shape: X={X_train.shape}, Y={Y_train.shape}")
    
    # 2. Instantiate model
    model = ChessTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # 3. Fit data
    print("\nRunning sanity check loop...")
    for epoch in range(50):
        loss = train_epoch(model, (X_train, Y_train), criterion, optimizer, device)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/50], Loss: {loss:.4f}")

if __name__ == "__main__":
    main()