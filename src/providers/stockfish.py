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

    def choose(self, board: chess.Board) -> dict:
        self.start()
        result = self.engine.play(board, chess.engine.Limit(depth=self.depth))
        move = result.move
        return {
            "move": move,
            "uci": move.uci(),
            "san": board.san(move),
            "provider": "Stockfish",
            "depth": self.depth,
        }

    def close(self):
        if self.engine is not None:
            self.engine.quit()
            self.engine = None

    def __enter__(self):
        return self.start()

    def __exit__(self, _exc_type, _exc, _tb):
        self.close()
