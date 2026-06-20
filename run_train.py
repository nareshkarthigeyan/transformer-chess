import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data_loader import ChessPGNDataset
from src.model import ChessTransformer
from src.train import train_one_epoch, save_checkpoint

def main():
    # Force Apple Silicon GPU usage
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    # --- Hyperparameters ---
    DATA_DIR = "data"               # Directory containing your 35 PGN files
    MAX_GAMES_PER_FILE = 20        # Caps games per file so your Mac memory stays happy
    BATCH_SIZE = 512                # Increased batch size for faster MPS processing
    LR = 5e-4
    EPOCHS = 10
    # -----------------------

    # 1. Load the Consolidated Data
    dataset = ChessPGNDataset(DATA_DIR, max_games_per_file=MAX_GAMES_PER_FILE)
    
    if len(dataset) == 0:
        print("Dataset is empty! Double check your 'data' folder has valid .pgn files.")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 2. Setup Architecture
    model = ChessTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # 3. Training Loop
    print("\nCommencing Consolidated Training Sequence...")
    for epoch in range(EPOCHS):
        avg_loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{EPOCHS}] -> Average Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 2 == 0:
            save_checkpoint(model, optimizer, epoch+1, avg_loss)

    print("\n Training complete! Ready for evaluation.")

if __name__ == "__main__":
    main()