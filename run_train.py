import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data_loader import (
    DEFAULT_DATASET_PATH,
    ChessNumpyDataset,
    build_stockfish_distilled_dataset,
)
from src.model import ChessTransformer
from src.train import train_one_epoch, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train the chess transformer.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--max-games-per-file", type=int, default=None)
    parser.add_argument("--stockfish-path", default=os.environ.get("STOCKFISH_PATH"))
    parser.add_argument("--stockfish-depth", type=int, default=15)
    parser.add_argument(
        "--stockfish-time",
        type=float,
        default=None,
        help="Seconds per Stockfish evaluation. Overrides --stockfish-depth.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--value-loss-weight", type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()

    # Force Apple Silicon GPU usage
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    if args.rebuild_dataset or not os.path.exists(args.dataset_path):
        build_stockfish_distilled_dataset(
            data_dir=args.data_dir,
            output_path=args.dataset_path,
            max_games_per_file=args.max_games_per_file,
            stockfish_path=args.stockfish_path,
            stockfish_depth=args.stockfish_depth,
            stockfish_time=args.stockfish_time,
        )

    # 1. Load the cached Stockfish-distilled NumPy data.
    dataset = ChessNumpyDataset(args.dataset_path)
    metadata = dataset.metadata
    if metadata:
        print(
            "Loaded cached dataset: "
            f"{len(dataset)} positions | "
            f"teacher depth={metadata.get('stockfish_depth')} | "
            f"teacher time={metadata.get('stockfish_time')}"
        )

    if len(dataset) == 0:
        print("Dataset is empty! Double check your 'data' folder has valid .pgn files.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=len(dataset) >= args.batch_size,
    )

    # 2. Setup Architecture
    model = ChessTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # 3. Training Loop
    print("\nCommencing Stockfish-distilled AlphaZero-style training sequence...")
    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(
            model,
            dataloader,
            optimizer,
            criterion,
            device,
            value_criterion=value_criterion,
            value_loss_weight=args.value_loss_weight,
        )
        print(f"Epoch [{epoch+1}/{args.epochs}] -> Average Loss: {avg_loss:.4f}")

        if (epoch + 1) % 2 == 0:
            save_checkpoint(model, optimizer, epoch+1, avg_loss)

    print("\nTraining complete. Ready for evaluation.")


if __name__ == "__main__":
    main()
