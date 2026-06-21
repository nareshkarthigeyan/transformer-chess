import os
import glob
import chess
import chess.pgn
import torch
from torch.utils.data import Dataset

PIECE_TO_TOKEN = {
    None: 0,
    chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3, 
    chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6
}

def board_to_sequence(board: chess.Board) -> torch.Tensor:
    sequence = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            sequence.append(0)
        else:
            token = PIECE_TO_TOKEN[piece.piece_type]
            if piece.color == chess.BLACK:
                token += 6
            sequence.append(token)
    return torch.tensor(sequence, dtype=torch.long)

def move_to_id(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square

class ChessPGNDataset(Dataset):
    def __init__(self, data_dir: str, max_games_per_file: int = 1000):
        """
        Scans a directory for all .pgn files and builds a consolidated dataset.
        """
        self.X = []
        self.Y = []
        self.fens = []  # Added to track FENs for Pillar 2 masking
        
        # Find all .pgn files in the directory
        pgn_files = glob.glob(os.path.join(data_dir, "*.pgn"))
        print(f"Found {len(pgn_files)} PGN files to process.")
        
        for file_path in pgn_files:
            file_name = os.path.basename(file_path)
            print(f"Processing: {file_name}...")
            
            with open(file_path, "r", encoding="utf-8", errors="ignore") as pgn:
                game_count = 0
                while game_count < max_games_per_file:
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        break # End of this specific file
                    
                    board = game.board()
                    for move in game.mainline_moves():
                        if move in board.legal_moves:
                            self.X.append(board_to_sequence(board))
                            self.Y.append(move_to_id(move))
                            self.fens.append(board.fen())  # Save current FEN before pushing move
                            board.push(move)
                    
                    game_count += 1
                    
            print(f" Loaded {game_count} games from {file_name}")
                    
        print("\nAll files consolidated.")
        print(f"Total board states in dataset: {len(self.X)}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Simply pull from your clean arrays (tensors are built inside __init__ or on the fly)
        board_seq = self.X[idx]
        move_id = self.Y[idx]
        fen = self.fens[idx]

        # Handle formatting conversions safely
        if not isinstance(board_seq, torch.Tensor):
            board_seq = torch.tensor(board_seq, dtype=torch.long)
        if not isinstance(move_id, torch.Tensor):
            move_id = torch.tensor(move_id, dtype=torch.long)

        return board_seq, move_id, fen