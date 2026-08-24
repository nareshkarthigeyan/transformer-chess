import os
import glob
import json
import shutil
from dataclasses import dataclass
from typing import Optional

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
from torch.utils.data import Dataset

PIECE_TO_TOKEN = {
    None: 0,
    chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3, 
    chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6
}

MAX_LEGAL_MOVES = 256
DEFAULT_DATASET_PATH = os.path.join("data", "stockfish_distilled_dataset.npz")


@dataclass(frozen=True)
class StockfishConfig:
    path: str = "/opt/homebrew/bin/stockfish"
    depth: Optional[int] = 15
    time_limit: Optional[float] = None
    centipawn_scale: int = 600


def board_to_array(board: chess.Board) -> np.ndarray:
    sequence = np.zeros(64, dtype=np.uint8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        token = PIECE_TO_TOKEN[piece.piece_type]
        if piece.color == chess.BLACK:
            token += 6
        sequence[square] = token
    return sequence


def board_to_sequence(board: chess.Board) -> torch.Tensor:
    return torch.from_numpy(board_to_array(board)).long()


def move_to_id(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


def _resolve_stockfish_path(path: Optional[str]) -> str:
    if path:
        return path
    return shutil.which("stockfish") or "/opt/homebrew/bin/stockfish"


def _engine_limit(config: StockfishConfig) -> chess.engine.Limit:
    if config.time_limit is not None:
        return chess.engine.Limit(time=config.time_limit)
    return chess.engine.Limit(depth=config.depth)


def _score_to_value(score: chess.engine.PovScore, board: chess.Board, scale: int) -> float:
    pov_score = score.pov(board.turn)
    mate = pov_score.mate()
    if mate is not None:
        return 1.0 if mate > 0 else -1.0

    centipawns = pov_score.score(mate_score=scale * 10)
    if centipawns is None:
        return 0.0
    return float(np.tanh(centipawns / scale))


def _legal_move_ids(board: chess.Board) -> np.ndarray:
    ids = np.full(MAX_LEGAL_MOVES, -1, dtype=np.int16)
    for idx, move in enumerate(board.legal_moves):
        if idx >= MAX_LEGAL_MOVES:
            raise ValueError(f"Position has more than {MAX_LEGAL_MOVES} legal moves.")
        ids[idx] = move_to_id(move)
    return ids


def _analyse_position(engine, board: chess.Board, config: StockfishConfig):
    info = engine.analyse(board, _engine_limit(config), multipv=1)
    if isinstance(info, list):
        info = info[0] if info else {}
    pv = info.get("pv") or []
    teacher_move = pv[0] if pv else None
    if teacher_move is None or teacher_move not in board.legal_moves:
        teacher_move = next(iter(board.legal_moves), None)

    score = info.get("score")
    value = _score_to_value(score, board, config.centipawn_scale) if score else 0.0
    return teacher_move, value


def build_stockfish_distilled_dataset(
    data_dir: str,
    output_path: str = DEFAULT_DATASET_PATH,
    max_games_per_file: Optional[int] = None,
    stockfish_path: Optional[str] = None,
    stockfish_depth: Optional[int] = 15,
    stockfish_time: Optional[float] = None,
    progress_interval: int = 500,
):
    """
    Converts all PGNs into a cached NumPy dataset with Stockfish policy/value labels.
    The policy target is the teacher's best legal move; the value is from the
    side-to-move perspective in [-1, 1].
    """
    engine_path = _resolve_stockfish_path(stockfish_path)
    if not os.path.exists(engine_path):
        raise FileNotFoundError(
            f"Stockfish binary not found at {engine_path}. "
            "Set STOCKFISH_PATH or pass --stockfish-path."
        )

    config = StockfishConfig(
        path=engine_path,
        depth=stockfish_depth,
        time_limit=stockfish_time,
    )

    pgn_files = sorted(glob.glob(os.path.join(data_dir, "*.pgn")))
    if not pgn_files:
        raise FileNotFoundError(f"No .pgn files found in {data_dir}.")

    boards = []
    human_move_ids = []
    teacher_move_ids = []
    values = []
    legal_move_ids = []
    fens = []
    source_files = []

    print(f"Found {len(pgn_files)} PGN files to process.")
    print(
        "Stockfish teacher: "
        f"{engine_path} | "
        f"{'time=' + str(stockfish_time) + 's' if stockfish_time is not None else 'depth=' + str(stockfish_depth)}"
    )

    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        for file_path in pgn_files:
            file_name = os.path.basename(file_path)
            game_count = 0
            print(f"Processing: {file_name}...")

            with open(file_path, "r", encoding="utf-8", errors="ignore") as pgn:
                while max_games_per_file is None or game_count < max_games_per_file:
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        break

                    board = game.board()
                    for move in game.mainline_moves():
                        if move not in board.legal_moves:
                            continue

                        teacher_move, value = _analyse_position(engine, board, config)
                        if teacher_move is None:
                            continue

                        boards.append(board_to_array(board))
                        human_move_ids.append(move_to_id(move))
                        teacher_move_ids.append(move_to_id(teacher_move))
                        values.append(value)
                        legal_move_ids.append(_legal_move_ids(board))
                        fens.append(board.fen())
                        source_files.append(file_name)

                        if len(boards) % progress_interval == 0:
                            print(f"   evaluated {len(boards)} positions...")

                        board.push(move)

                    game_count += 1

            print(f" Loaded {game_count} games from {file_name}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    metadata = {
        "data_dir": data_dir,
        "pgn_files": [os.path.basename(path) for path in pgn_files],
        "stockfish_path": engine_path,
        "stockfish_depth": stockfish_depth,
        "stockfish_time": stockfish_time,
        "centipawn_scale": config.centipawn_scale,
        "positions": len(boards),
    }

    np.savez_compressed(
        output_path,
        boards=np.asarray(boards, dtype=np.uint8),
        human_move_ids=np.asarray(human_move_ids, dtype=np.int16),
        teacher_move_ids=np.asarray(teacher_move_ids, dtype=np.int16),
        values=np.asarray(values, dtype=np.float32),
        legal_move_ids=np.asarray(legal_move_ids, dtype=np.int16),
        fens=np.asarray(fens, dtype=np.str_),
        source_files=np.asarray(source_files, dtype=np.str_),
        metadata=np.asarray(json.dumps(metadata), dtype=np.str_),
    )

    print("\nAll files consolidated and evaluated.")
    print(f"Total board states in dataset: {len(boards)}")
    print(f"Cached NumPy dataset written to: {output_path}")
    return output_path


class ChessNumpyDataset(Dataset):
    def __init__(self, dataset_path: str = DEFAULT_DATASET_PATH):
        self.dataset_path = dataset_path
        self.data = np.load(dataset_path, allow_pickle=False)
        self.X = self.data["boards"]
        self.policy_targets = self.data["teacher_move_ids"]
        self.values = self.data["values"]
        self.legal_move_ids = self.data["legal_move_ids"]

    @property
    def metadata(self):
        if "metadata" not in self.data:
            return {}
        return json.loads(str(self.data["metadata"].item()))

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return (
            torch.as_tensor(self.X[idx], dtype=torch.long),
            torch.as_tensor(self.policy_targets[idx], dtype=torch.long),
            torch.as_tensor(self.values[idx], dtype=torch.float32),
            torch.as_tensor(self.legal_move_ids[idx], dtype=torch.long),
        )


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
