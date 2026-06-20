import math
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Optional

import chess
import chess.engine

from .evaluate import get_best_move

NUM_GAMES = 20
MAX_MOVES = 150
RANDOM_BASE_ELO = 100
STOCKFISH_DEPTH_1_BASE_ELO = 600
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"

ShouldStopCallback = Callable[[], bool]
MoveCallback = Callable[[chess.Board, chess.Move, str], None]


@dataclass
class GameResult:
    board: chess.Board
    model_color: chess.Color
    move_count: int
    result: str
    reason: str
    category: str


def get_elo_from_percentage(percentage: float) -> int:
    if percentage >= 100:
        return 400
    if percentage <= 0:
        return -400
    return int(-400 * math.log10((100 - percentage) / percentage))


def summarize_results(
    wins: int,
    losses: int,
    draws: int,
    opponent_type: str,
    num_games: int = NUM_GAMES,
) -> dict:
    total_score = wins + (0.5 * draws)
    win_pct = (total_score / num_games) * 100 if num_games else 0
    base_elo = (
        STOCKFISH_DEPTH_1_BASE_ELO
        if opponent_type == "stockfish_depth_1"
        else RANDOM_BASE_ELO
    )
    calculated_elo = base_elo + get_elo_from_percentage(win_pct)
    return {
        "W": wins,
        "L": losses,
        "D": draws,
        "Win%": win_pct,
        "Elo": calculated_elo,
    }


def open_stockfish(opponent_type: str, stockfish_path: str = STOCKFISH_PATH):
    if opponent_type != "stockfish_depth_1":
        return None
    if not os.path.exists(stockfish_path):
        raise FileNotFoundError(f"Stockfish binary not found at {stockfish_path}.")
    return chess.engine.SimpleEngine.popen_uci(stockfish_path)


def choose_opponent_move(board: chess.Board, opponent_type: str, engine=None) -> chess.Move:
    if opponent_type == "stockfish_depth_1" and engine is not None:
        result = engine.play(board, chess.engine.Limit(time=0.001, depth=1))
        return result.move
    return random.choice(list(board.legal_moves))


def categorize_result(result: str, model_color: chess.Color) -> str:
    if result == "1/2-1/2":
        return "draw"
    if result == "1-0":
        return "win" if model_color == chess.WHITE else "lose"
    if result == "0-1":
        return "win" if model_color == chess.BLACK else "lose"
    return "draw"


def play_single_game(
    model,
    device,
    opponent_type: str,
    model_color: chess.Color,
    engine=None,
    max_moves: int = MAX_MOVES,
    should_stop: Optional[ShouldStopCallback] = None,
    on_move: Optional[MoveCallback] = None,
    move_delay: float = 0,
) -> GameResult:
    board = chess.Board()
    move_count = 0
    forfeit = False

    while not board.is_game_over() and move_count < max_moves:
        if should_stop and should_stop():
            break

        if board.turn == model_color:
            move = get_best_move(model, board, device)
            mover = "Model"
            if move is None or move not in board.legal_moves:
                forfeit = True
                break
        else:
            move = choose_opponent_move(board, opponent_type, engine)
            mover = "Opponent"

        board.push(move)
        move_count += 1

        if on_move:
            on_move(board, move, mover)
        if move_delay > 0:
            time.sleep(move_delay)

    if forfeit:
        result = "1-0" if model_color == chess.BLACK else "0-1"
        reason = "Illegal Move Forfeit"
    elif move_count >= max_moves:
        result = "1/2-1/2"
        reason = "Max Move Cap"
    else:
        result = board.result()
        if board.is_checkmate():
            reason = f"Checkmate by {'Opponent' if board.turn == model_color else 'Model'}"
        else:
            reason = "Draw by Rules"

    category = categorize_result(result, model_color)
    return GameResult(
        board=board,
        model_color=model_color,
        move_count=move_count,
        result=result,
        reason=reason,
        category=category,
    )
