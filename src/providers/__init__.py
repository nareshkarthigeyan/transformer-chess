"""Move providers used by the playable dashboard.

The transformer remains the primary research model. Stockfish is isolated in
its own provider so it can be removed without touching the game UI or model
code.
"""

from .stockfish import StockfishProvider, resolve_stockfish_path
from .transformer import TransformerProvider

__all__ = ["StockfishProvider", "TransformerProvider", "resolve_stockfish_path"]
