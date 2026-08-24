import math

import chess
import torch
import torch.nn.functional as F

from ..data_loader import board_to_sequence, move_to_id


def _move_label(board: chess.Board, move: chess.Move) -> dict:
    return {"uci": move.uci(), "san": board.san(move)}


class TransformerProvider:
    """Inference plus a compact, real logit-lens readout for each encoder layer."""

    name = "transformer"

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def analyze(self, board: chess.Board, top_k: int = 4) -> dict:
        self.model.eval()
        board_tensor = board_to_sequence(board).unsqueeze(0).to(self.device)
        legal_moves = list(board.legal_moves)
        legal_ids = torch.tensor(
            [move_to_id(move) for move in legal_moves],
            dtype=torch.long,
            device=self.device,
        )

        if not legal_moves:
            return {
                "provider": "Transformer",
                "selected_move": None,
                "layers": [],
                "value": 0.0,
                "entropy": 0.0,
            }

        with torch.no_grad():
            embedded = self.model._embed_board(board_tensor)
            final_stream, activations = self.model.encode_with_intermediates(board_tensor)
            streams = [("Input projection", embedded)]
            streams.extend(
                (f"Encoder {index + 1}", activation)
                for index, activation in enumerate(activations)
            )

            layer_results = [
                self._project_layer(board, stream, legal_moves, legal_ids, top_k, name)
                for name, stream in streams
            ]
            final_stream_flat = final_stream.contiguous().view(1, -1)
            final_value = float(self.model.value_head(final_stream_flat).item())

        final = layer_results[-1]
        return {
            "provider": "Transformer",
            "selected_uci": final["move"].uci(),
            "selected_san": final["san"],
            "value": round(final_value, 4),
            "entropy": final["entropy"],
            "layers": [
                {key: value for key, value in layer.items() if key != "move"}
                for layer in layer_results
            ],
            "top_moves": final["top_moves"],
        }

    def _project_layer(
        self,
        board,
        stream,
        legal_moves,
        legal_ids,
        top_k,
        name,
    ):
        flat = stream.contiguous().view(1, -1)
        logits = self.model.fc_out(flat).squeeze(0)
        legal_logits = logits.index_select(0, legal_ids)
        probabilities = F.softmax(legal_logits, dim=0)
        entropy = float(
            (-(probabilities * probabilities.clamp_min(1e-9).log()).sum())
            / math.log(max(2, len(legal_moves)))
        )
        order = torch.argsort(probabilities, descending=True)
        top_moves = []
        for index in order[:top_k].tolist():
            move = legal_moves[index]
            top_moves.append(
                {
                    **_move_label(board, move),
                    "probability": round(float(probabilities[index].item()), 4),
                }
            )

        best_index = int(order[0].item())
        best_move = legal_moves[best_index]
        value = float(self.model.value_head(flat).item())
        return {
            "name": name,
            "move": best_move,
            "san": board.san(best_move),
            "confidence": round(float(probabilities[best_index].item()), 4),
            "entropy": round(entropy, 4),
            "value": round(value, 4),
            "top_moves": top_moves,
        }
