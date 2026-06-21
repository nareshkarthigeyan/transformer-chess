import io

import chess
import chess.pgn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_loader import board_to_sequence, move_to_id
from src.model import ChessTransformer
from src.train import train_one_epoch


def process_pgn_games(pgn_text):
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        raise ValueError("Sample PGN did not contain a readable game.")

    board = game.board()
    x_rows = []
    y_rows = []
    fens = []

    for move in game.mainline_moves():
        if move not in board.legal_moves:
            continue
        x_rows.append(board_to_sequence(board))
        y_rows.append(move_to_id(move))
        fens.append(board.fen())
        board.push(move)

    if not x_rows:
        raise ValueError("Sample PGN did not produce any legal training rows.")

    return torch.stack(x_rows), torch.tensor(y_rows, dtype=torch.long), fens


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    sample_pgn = "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 1-0"
    X_train, Y_train, fens = process_pgn_games(sample_pgn)
    print(f"Processed shape: X={X_train.shape}, Y={Y_train.shape}")

    dataloader = DataLoader(
        list(zip(X_train, Y_train, fens)),
        batch_size=len(X_train),
        shuffle=True,
    )

    model = ChessTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("\nRunning sanity check loop...")
    for epoch in range(50):
        loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/50], Loss: {loss:.4f}")


if __name__ == "__main__":
    main()
