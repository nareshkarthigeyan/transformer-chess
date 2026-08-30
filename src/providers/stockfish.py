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

    def close(self):
        if self.engine is not None:
            self.engine.quit()
            self.engine = None

    def __enter__(self):
        return self.start()

    def __exit__(self, _exc_type, _exc, _tb):
        self.close()
