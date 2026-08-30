"""PGN ingestion and resumable Stockfish policy/value distillation."""

from __future__ import annotations

import glob
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
from torch.utils.data import Dataset

PIECE_TO_TOKEN = {
    None: 0,
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}

MAX_LEGAL_MOVES = 256
DEFAULT_DATASET_PATH = os.path.join("data", "stockfish_distilled_dataset.npz")
DATASET_VERSION = 2


@dataclass(frozen=True)
class StockfishConfig:
    path: str
    depth: Optional[int] = 15
    time_limit: Optional[float] = None
    centipawn_scale: int = 600
    multipv: int = 4
    policy_temperature: int = 120


def board_to_array(board: chess.Board) -> np.ndarray:
    """Encode the 64 squares as piece IDs from White's perspective."""
    sequence = np.zeros(64, dtype=np.uint8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            token = PIECE_TO_TOKEN[piece.piece_type]
            sequence[square] = token + (6 if piece.color == chess.BLACK else 0)
    return sequence


def board_to_state_features(board: chess.Board) -> np.ndarray:
    """Encode state that cannot be recovered from piece placement alone.

    The fixed-width feature vector contains side-to-move, castling rights,
    en-passant square (+1 reserves 0 for no square), and bounded clocks.
    """
    rights = 0
    rights |= int(board.has_kingside_castling_rights(chess.WHITE)) << 0
    rights |= int(board.has_queenside_castling_rights(chess.WHITE)) << 1
    rights |= int(board.has_kingside_castling_rights(chess.BLACK)) << 2
    rights |= int(board.has_queenside_castling_rights(chess.BLACK)) << 3
    ep_square = 0 if board.ep_square is None else board.ep_square + 1
    return np.asarray(
        [
            int(board.turn == chess.BLACK),
            rights,
            ep_square,
            min(board.halfmove_clock, 100),
            min(board.fullmove_number, 255),
        ],
        dtype=np.uint16,
    )


def board_to_sequence(board: chess.Board) -> torch.Tensor:
    return torch.from_numpy(board_to_array(board)).long()


def board_to_state_tensor(board: chess.Board) -> torch.Tensor:
    return torch.from_numpy(board_to_state_features(board)).long()


def move_to_id(move: chess.Move) -> int:
    """Map source/destination to the policy head's 4,096 move buckets.

    Legal-move selection is still done over full python-chess Move objects, so
    every played move is legal. Queen promotion is preferred when promotion
    variants share a source/destination bucket; underpromotions remain a known
    limitation of this compact 4,096-output architecture and are reported in
    the project documentation rather than silently treated as different IDs.
    """
    return move.from_square * 64 + move.to_square


def _resolve_stockfish_path(path: Optional[str]) -> str:
    candidates = [
        path,
        os.environ.get("STOCKFISH_PATH"),
        shutil.which("stockfish"),
        shutil.which("stockfish.exe"),
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/usr/games/stockfish",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates if candidate)
    raise FileNotFoundError(
        "Stockfish binary was not found. Install it, set STOCKFISH_PATH, or "
        f"pass --stockfish-path. Checked: {searched or 'PATH'}"
    )


def _engine_limit(config: StockfishConfig) -> chess.engine.Limit:
    if config.time_limit is not None:
        return chess.engine.Limit(time=config.time_limit)
    return chess.engine.Limit(depth=config.depth)


def _score_to_centipawns(score: chess.engine.PovScore, board: chess.Board, scale: int) -> int:
    pov_score = score.pov(board.turn)
    mate = pov_score.mate()
    if mate is not None:
        return (scale * 10) if mate > 0 else -(scale * 10)
    return int(pov_score.score(mate_score=scale * 10) or 0)


def _score_to_value(score: chess.engine.PovScore, board: chess.Board, scale: int) -> float:
    return float(np.tanh(_score_to_centipawns(score, board, scale) / scale))


def _legal_move_ids(board: chess.Board) -> np.ndarray:
    ids = np.full(MAX_LEGAL_MOVES, -1, dtype=np.int16)
    for index, move in enumerate(board.legal_moves):
        if index >= MAX_LEGAL_MOVES:
            raise ValueError(f"Position has more than {MAX_LEGAL_MOVES} legal moves.")
        ids[index] = move_to_id(move)
    return ids


def _analyse_position(engine, board: chess.Board, config: StockfishConfig):
    """Return top teacher move, side-to-move value, and a soft top-k policy."""
    infos = engine.analyse(board, _engine_limit(config), multipv=config.multipv)
    if not isinstance(infos, list):
        infos = [infos]

    candidates = []
    for info in infos:
        pv = info.get("pv") or []
        move = pv[0] if pv else None
        if move is None or move not in board.legal_moves:
            continue
        score = info.get("score")
        centipawns = _score_to_centipawns(score, board, config.centipawn_scale) if score else 0
        candidates.append((move, centipawns, score))

    if not candidates:
        fallback = next(iter(board.legal_moves), None)
        if fallback is None:
            return None, 0.0, [], []
        candidates = [(fallback, 0, None)]

    candidate_scores = np.asarray([entry[1] for entry in candidates], dtype=np.float64)
    scaled = (candidate_scores - candidate_scores.max()) / max(1, config.policy_temperature)
    probabilities = np.exp(scaled)
    probabilities = probabilities / probabilities.sum()
    root_score = candidates[0][2]
    value = _score_to_value(root_score, board, config.centipawn_scale) if root_score else 0.0
    return candidates[0][0], value, [move_to_id(entry[0]) for entry in candidates], probabilities.tolist()


def _partial_path(output_path: str) -> str:
    root, suffix = os.path.splitext(output_path)
    return f"{root}.partial{suffix or '.npz'}"


def _save_dataset(path: str, rows: dict, metadata: dict) -> None:
    """Write atomically so a Colab/Kaggle disconnect cannot corrupt a cache."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary_path = f"{path}.tmp.npz"
    np.savez_compressed(
        temporary_path,
        boards=np.asarray(rows["boards"], dtype=np.uint8),
        state_features=np.asarray(rows["state_features"], dtype=np.uint16),
        human_move_ids=np.asarray(rows["human_move_ids"], dtype=np.int16),
        teacher_move_ids=np.asarray(rows["teacher_move_ids"], dtype=np.int16),
        values=np.asarray(rows["values"], dtype=np.float32),
        legal_move_ids=np.asarray(rows["legal_move_ids"], dtype=np.int16),
        teacher_top_move_ids=np.asarray(rows["teacher_top_move_ids"], dtype=np.int16),
        teacher_top_probs=np.asarray(rows["teacher_top_probs"], dtype=np.float32),
        fens=np.asarray(rows["fens"], dtype=np.str_),
        source_files=np.asarray(rows["source_files"], dtype=np.str_),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    os.replace(temporary_path, path)


def _load_partial(path: str) -> tuple[dict, dict]:
    loaded = np.load(path, allow_pickle=False)
    metadata = json.loads(str(loaded["metadata"].item()))
    required = [
        "boards", "state_features", "human_move_ids", "teacher_move_ids", "values",
        "legal_move_ids", "teacher_top_move_ids", "teacher_top_probs", "fens", "source_files",
    ]
    if any(key not in loaded for key in required):
        raise ValueError("Partial dataset was created by an incompatible pipeline version.")
    return ({key: loaded[key].tolist() for key in required}, metadata)


def build_stockfish_distilled_dataset(
    data_dir: str,
    output_path: str = DEFAULT_DATASET_PATH,
    max_games_per_file: Optional[int] = None,
    stockfish_path: Optional[str] = None,
    stockfish_depth: Optional[int] = 15,
    stockfish_time: Optional[float] = None,
    progress_interval: int = 500,
    max_positions: Optional[int] = None,
    sample_every: int = 1,
    min_ply: int = 0,
    max_ply: Optional[int] = None,
    teacher_multipv: int = 4,
    policy_temperature: int = 120,
    stockfish_threads: int = 2,
    stockfish_hash_mb: int = 256,
    resume: bool = True,
    checkpoint_interval: int = 2_000,
):
    """Convert PGNs into a cached, resumable Stockfish-distilled dataset.

    The cache contains exact legal move IDs for masking, a hard best-move
    target, a top-k soft teacher policy, and a Stockfish side-to-move value.
    A ``.partial.npz`` sibling is safely refreshed while labelling; running the
    same command again continues at the next selected position.
    """
    if sample_every < 1:
        raise ValueError("sample_every must be at least 1")
    if max_positions is not None and max_positions < 1:
        raise ValueError("max_positions must be positive")
    if teacher_multipv < 1:
        raise ValueError("teacher_multipv must be at least 1")

    engine_path = _resolve_stockfish_path(stockfish_path)
    pgn_files = sorted(glob.glob(os.path.join(data_dir, "*.pgn")))
    if not pgn_files:
        raise FileNotFoundError(f"No .pgn files found in {data_dir}.")

    config = StockfishConfig(
        path=engine_path,
        depth=stockfish_depth,
        time_limit=stockfish_time,
        multipv=teacher_multipv,
        policy_temperature=policy_temperature,
    )
    partial_path = _partial_path(output_path)
    top_k = teacher_multipv
    rows = {
        "boards": [], "state_features": [], "human_move_ids": [], "teacher_move_ids": [],
        "values": [], "legal_move_ids": [], "teacher_top_move_ids": [],
        "teacher_top_probs": [], "fens": [], "source_files": [],
    }
    skipped_selected_positions = 0
    if resume and os.path.isfile(partial_path):
        rows, partial_metadata = _load_partial(partial_path)
        if partial_metadata.get("dataset_version") != DATASET_VERSION:
            raise ValueError("Partial dataset version mismatch; delete it or use --no-resume.")
        skipped_selected_positions = len(rows["boards"])
        print(f"Resuming partial distillation: {skipped_selected_positions:,} labelled positions.")
    elif os.path.isfile(partial_path):
        os.remove(partial_path)

    metadata = {
        "dataset_version": DATASET_VERSION,
        "data_dir": data_dir,
        "pgn_files": [os.path.basename(path) for path in pgn_files],
        "stockfish_path": engine_path,
        "stockfish_depth": stockfish_depth,
        "stockfish_time": stockfish_time,
        "teacher_multipv": teacher_multipv,
        "policy_temperature": policy_temperature,
        "sample_every": sample_every,
        "min_ply": min_ply,
        "max_ply": max_ply,
        "max_positions": max_positions,
        "centipawn_scale": config.centipawn_scale,
    }
    print(f"Found {len(pgn_files)} PGN files. Stockfish: {engine_path}")
    print(
        "Teacher budget: "
        f"{'time=' + str(stockfish_time) + 's' if stockfish_time is not None else 'depth=' + str(stockfish_depth)} "
        f"| MultiPV={teacher_multipv} | sample every {sample_every} ply"
    )

    selected_seen = 0
    interrupted = False
    try:
        with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
            try:
                engine.configure({"Threads": max(1, stockfish_threads), "Hash": max(16, stockfish_hash_mb)})
            except chess.engine.EngineError:
                print("Stockfish did not accept Threads/Hash options; continuing with its defaults.")

            for file_path in pgn_files:
                if max_positions is not None and len(rows["boards"]) >= max_positions:
                    break
                file_name = os.path.basename(file_path)
                game_count = 0
                print(f"Processing {file_name}…")
                with open(file_path, "r", encoding="utf-8", errors="ignore") as pgn:
                    while max_games_per_file is None or game_count < max_games_per_file:
                        game = chess.pgn.read_game(pgn)
                        if game is None:
                            break
                        board = game.board()
                        for ply, human_move in enumerate(game.mainline_moves()):
                            if human_move not in board.legal_moves:
                                break
                            should_label = (
                                ply >= min_ply
                                and (max_ply is None or ply <= max_ply)
                                and ((ply - min_ply) % sample_every == 0)
                            )
                            if should_label:
                                if selected_seen < skipped_selected_positions:
                                    selected_seen += 1
                                elif max_positions is None or len(rows["boards"]) < max_positions:
                                    teacher_move, value, top_ids, top_probs = _analyse_position(engine, board, config)
                                    if teacher_move is not None:
                                        padded_ids = np.full(top_k, -1, dtype=np.int16)
                                        padded_probs = np.zeros(top_k, dtype=np.float32)
                                        count = min(top_k, len(top_ids))
                                        padded_ids[:count] = top_ids[:count]
                                        padded_probs[:count] = top_probs[:count]
                                        rows["boards"].append(board_to_array(board))
                                        rows["state_features"].append(board_to_state_features(board))
                                        rows["human_move_ids"].append(move_to_id(human_move))
                                        rows["teacher_move_ids"].append(move_to_id(teacher_move))
                                        rows["values"].append(value)
                                        rows["legal_move_ids"].append(_legal_move_ids(board))
                                        rows["teacher_top_move_ids"].append(padded_ids)
                                        rows["teacher_top_probs"].append(padded_probs)
                                        rows["fens"].append(board.fen())
                                        rows["source_files"].append(file_name)
                                        selected_seen += 1
                                        count_rows = len(rows["boards"])
                                        if count_rows % progress_interval == 0:
                                            print(f"  labelled {count_rows:,} positions…")
                                        if count_rows % checkpoint_interval == 0:
                                            metadata["positions"] = count_rows
                                            metadata["status"] = "partial"
                                            _save_dataset(partial_path, rows, metadata)
                                else:
                                    break
                            board.push(human_move)
                        game_count += 1
                print(f"  read {game_count:,} games from {file_name}")
    except KeyboardInterrupt:
        interrupted = True
        print("Distillation interrupted; saving a resumable partial cache.")
    finally:
        metadata["positions"] = len(rows["boards"])
        metadata["status"] = "partial" if interrupted else "complete"
        if rows["boards"]:
            _save_dataset(partial_path, rows, metadata)

    if interrupted:
        return partial_path
    if not rows["boards"]:
        raise RuntimeError("No valid positions were labelled from the supplied PGNs.")
    metadata["status"] = "complete"
    _save_dataset(output_path, rows, metadata)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    print(f"Distillation complete: {len(rows['boards']):,} positions → {output_path}")
    return output_path


class ChessNumpyDataset(Dataset):
    """Memory-mapped-style access to the NumPy cache produced above."""

    def __init__(self, dataset_path: str = DEFAULT_DATASET_PATH):
        self.dataset_path = dataset_path
        self.data = np.load(dataset_path, allow_pickle=False)
        self.X = self.data["boards"]
        self.policy_targets = self.data["teacher_move_ids"]
        self.values = self.data["values"]
        self.legal_move_ids = self.data["legal_move_ids"]
        self.teacher_top_move_ids = self.data.get("teacher_top_move_ids")
        self.teacher_top_probs = self.data.get("teacher_top_probs")
        if "state_features" in self.data:
            self.state_features = self.data["state_features"]
        elif "fens" in self.data:
            # Old cache compatibility: derive state safely once at load time.
            self.state_features = np.asarray(
                [board_to_state_features(chess.Board(fen)) for fen in self.data["fens"]], dtype=np.uint16
            )
        else:
            self.state_features = np.zeros((len(self.X), 5), dtype=np.uint16)

    @property
    def metadata(self):
        if "metadata" not in self.data:
            return {}
        return json.loads(str(self.data["metadata"].item()))

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        # Seven entries retain tuple compatibility with the original train loop.
        if self.teacher_top_move_ids is None or self.teacher_top_probs is None:
            top_ids = torch.full((1,), -1, dtype=torch.long)
            top_probs = torch.zeros((1,), dtype=torch.float32)
        else:
            top_ids = torch.as_tensor(self.teacher_top_move_ids[idx], dtype=torch.long)
            top_probs = torch.as_tensor(self.teacher_top_probs[idx], dtype=torch.float32)
        return (
            torch.as_tensor(self.X[idx], dtype=torch.long),
            torch.as_tensor(self.state_features[idx], dtype=torch.long),
            torch.as_tensor(self.policy_targets[idx], dtype=torch.long),
            torch.as_tensor(self.values[idx], dtype=torch.float32),
            torch.as_tensor(self.legal_move_ids[idx], dtype=torch.long),
            top_ids,
            top_probs,
        )


class ChessPGNDataset(Dataset):
    """Legacy human-move dataset kept for quick experimentation and tests."""

    def __init__(self, data_dir: str, max_games_per_file: int = 1_000):
        self.X, self.states, self.Y, self.fens = [], [], [], []
        pgn_files = sorted(glob.glob(os.path.join(data_dir, "*.pgn")))
        print(f"Found {len(pgn_files)} PGN files to process.")
        for file_path in pgn_files:
            game_count = 0
            with open(file_path, "r", encoding="utf-8", errors="ignore") as pgn:
                while game_count < max_games_per_file:
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        break
                    board = game.board()
                    for move in game.mainline_moves():
                        if move not in board.legal_moves:
                            break
                        self.X.append(board_to_sequence(board))
                        self.states.append(board_to_state_tensor(board))
                        self.Y.append(move_to_id(move))
                        self.fens.append(board.fen())
                        board.push(move)
                    game_count += 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.states[idx], torch.tensor(self.Y[idx], dtype=torch.long), self.fens[idx]
