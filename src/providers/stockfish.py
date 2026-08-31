import math
import os
import shutil
from typing import Optional

import chess
import chess.engine


def resolve_stockfish_path(explicit_path: Optional[str] = None) -> Optional[str]:
    """Find Stockfish on macOS, Linux, or Windows without hard-coding one OS."""
    candidates = [
        explicit_path,
        os.environ.get("STOCKFISH_PATH"),
        shutil.which("stockfish"),
        shutil.which("stockfish.exe"),
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/usr/games/stockfish",
        r"C:\\Program Files\\Stockfish\\stockfish.exe",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


class StockfishProvider:
    """Small UCI adapter kept separate from the transformer provider."""

    name = "stockfish"

    def __init__(self, path: Optional[str] = None, depth: int = 10):
        self.path = resolve_stockfish_path(path)
        self.depth = depth
        self.engine = None

    @property
    def available(self) -> bool:
        return self.path is not None

    def start(self):
        if not self.available:
            raise FileNotFoundError(
                "Stockfish was not found. Install it or set STOCKFISH_PATH."
            )
        if self.engine is None:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.path)
        return self

    def choose(self, board: chess.Board, depth: Optional[int] = None) -> dict:
        self.start()
        search_depth = depth if depth is not None else self.depth
        result = self.engine.play(board, chess.engine.Limit(depth=search_depth))
        move = result.move
        return {
            "move": move,
            "uci": move.uci(),
            "san": board.san(move),
            "provider": "Stockfish",
            "depth": search_depth,
        }

    @staticmethod
    def _centipawns(score, perspective: chess.Color) -> int:
        return int(score.pov(perspective).score(mate_score=10_000) or 0)

    @staticmethod
    def _quality_band(centipawn_loss: int) -> str:
        if centipawn_loss <= 15:
            return "Excellent"
        if centipawn_loss <= 45:
            return "Good"
        if centipawn_loss <= 100:
            return "Inaccuracy"
        if centipawn_loss <= 250:
            return "Mistake"
        return "Blunder"

    @staticmethod
    def _heuristic_band(centipawn_loss: int) -> str:
        """A move-quality band, explicitly not a player Elo calculation."""
        if centipawn_loss <= 15:
            return "1800+ quality band"
        if centipawn_loss <= 45:
            return "1400–1800 quality band"
        if centipawn_loss <= 100:
            return "1000–1400 quality band"
        return "below 1000 quality band"

    def review_move(self, board: chess.Board, move: chess.Move) -> dict:
        """Analyse a played legal move using centipawn loss at fixed depth.

        A single move cannot reveal a person's Elo, so the returned ``quality``
        and ``heuristic_band`` are deliberately framed as local diagnostics.
        """
        if move not in board.legal_moves:
            raise ValueError("Cannot review an illegal move.")
        self.start()
        side = board.turn
        infos = self.engine.analyse(board, chess.engine.Limit(depth=self.depth), multipv=1)
        if isinstance(infos, list):
            infos = infos[0]
        best_move = (infos.get("pv") or [None])[0]
        best_score = self._centipawns(infos["score"], side) if infos.get("score") else 0
        played_san = board.san(move)
        after = board.copy(stack=False)
        after.push(move)
        played_info = self.engine.analyse(after, chess.engine.Limit(depth=self.depth), multipv=1)
        if isinstance(played_info, list):
            played_info = played_info[0]
        played_score = (
            self._centipawns(played_info["score"], side) if played_info.get("score") else best_score
        )
        centipawn_loss = max(0, best_score - played_score)
        return {
            "available": True,
            "depth": self.depth,
            "best_move": best_move.uci() if best_move else None,
            "best_san": board.san(best_move) if best_move else None,
            "played_san": played_san,
            "best_centipawns": best_score,
            "played_centipawns": played_score,
            "centipawn_loss": centipawn_loss,
            "quality": self._quality_band(centipawn_loss),
            "heuristic_band": self._heuristic_band(centipawn_loss),
            "note": "Heuristic move-quality band; it is not an official player Elo estimate.",
        }

    def evaluate_position(self, board: chess.Board, depth: Optional[int] = None) -> dict:
        """Return a live side-to-side evaluation for the current position.

        Scores are reported from White's perspective in centipawns.  Mate
        scores are kept separately so the UI can display ``#N`` instead of
        pretending a forced mate is an ordinary numeric advantage.
        """
        self.start()
        search_depth = depth if depth is not None else self.depth
        info = self.engine.analyse(board, chess.engine.Limit(depth=search_depth), multipv=1)
        if isinstance(info, list):
            info = info[0]
        score = info.get("score")
        white_score = score.pov(chess.WHITE) if score is not None else None
        mate = white_score.mate() if white_score is not None else None
        centipawns = int(white_score.score(mate_score=10_000) or 0) if white_score is not None else 0
        if mate is not None:
            white_wins = board.turn == chess.BLACK if mate == 0 and board.is_checkmate() else mate > 0
            white_win_probability = 0.999 if white_wins else 0.001
            display = f"#{abs(mate)}" if white_wins else f"-#{abs(mate)}"
        else:
            # Smoothly maps a centipawn score to a presentation-only chance
            # bar. This is not a win probability calibrated for tournament
            # play; it simply mirrors the side-to-side advantage visualization.
            white_win_probability = 1.0 / (1.0 + math.exp(-centipawns / 400.0))
            white_win_probability = min(0.999, max(0.001, white_win_probability))
            display = f"{centipawns / 100:+.2f}"
        return {
            "available": True,
            "depth": search_depth,
            "centipawns": centipawns,
            "mate": mate,
            "display": display,
            "white_win_probability": round(white_win_probability, 4),
            "side_to_move": "white" if board.turn == chess.WHITE else "black",
        }

    @staticmethod
    def _centipawns(score, perspective: chess.Color) -> int:
        return int(score.pov(perspective).score(mate_score=10_000) or 0)

    @staticmethod
    def _quality_band(centipawn_loss: int) -> str:
        if centipawn_loss <= 15:
            return "Excellent"
        if centipawn_loss <= 45:
            return "Good"
        if centipawn_loss <= 100:
            return "Inaccuracy"
        if centipawn_loss <= 250:
            return "Mistake"
        return "Blunder"

    @staticmethod
    def _heuristic_band(centipawn_loss: int) -> str:
        """A move-quality band, explicitly not a player Elo calculation."""
        if centipawn_loss <= 15:
            return "1800+ quality band"
        if centipawn_loss <= 45:
            return "1400–1800 quality band"
        if centipawn_loss <= 100:
            return "1000–1400 quality band"
        return "below 1000 quality band"

    def review_move(self, board: chess.Board, move: chess.Move) -> dict:
        """Analyse a played legal move using centipawn loss at fixed depth.

        A single move cannot reveal a person's Elo, so the returned ``quality``
        and ``heuristic_band`` are deliberately framed as local diagnostics.
        """
        if move not in board.legal_moves:
            raise ValueError("Cannot review an illegal move.")
        self.start()
        side = board.turn
        infos = self.engine.analyse(board, chess.engine.Limit(depth=self.depth), multipv=1)
        if isinstance(infos, list):
            infos = infos[0]
        best_move = (infos.get("pv") or [None])[0]
        best_score = self._centipawns(infos["score"], side) if infos.get("score") else 0
        played_san = board.san(move)
        after = board.copy(stack=False)
        after.push(move)
        played_info = self.engine.analyse(after, chess.engine.Limit(depth=self.depth), multipv=1)
        if isinstance(played_info, list):
            played_info = played_info[0]
        played_score = (
            self._centipawns(played_info["score"], side) if played_info.get("score") else best_score
        )
        centipawn_loss = max(0, best_score - played_score)
        return {
            "available": True,
            "depth": self.depth,
            "best_move": best_move.uci() if best_move else None,
            "best_san": board.san(best_move) if best_move else None,
            "played_san": played_san,
            "best_centipawns": best_score,
            "played_centipawns": played_score,
            "centipawn_loss": centipawn_loss,
            "quality": self._quality_band(centipawn_loss),
            "heuristic_band": self._heuristic_band(centipawn_loss),
            "note": "Heuristic move-quality band; it is not an official player Elo estimate.",
        }

    def close(self):
        if self.engine is not None:
            self.engine.quit()
            self.engine = None

    def __enter__(self):
        return self.start()

    def __exit__(self, _exc_type, _exc, _tb):
        self.close()
