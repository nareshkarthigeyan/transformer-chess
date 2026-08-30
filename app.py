import atexit
import math
import os
import random
from copy import deepcopy
from threading import Lock, Thread

import chess
import torch
from flask import Flask, jsonify, render_template, request

from src import (
    ChessTransformer,
    MAX_MOVES,
    NUM_GAMES,
    load_checkpoint_weights,
    open_stockfish,
    play_single_game,
    summarize_results,
)
from src.providers import StockfishProvider, TransformerProvider

app = Flask(__name__)


def _select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = _select_device()
checkpoint_path = os.environ.get("CHECKPOINT_PATH")
if checkpoint_path is None:
    checkpoint_path = "checkpoints/best.pt" if os.path.isfile("checkpoints/best.pt") else "checkpoint.pt"


def _checkpoint_model_config(path):
    if not os.path.isfile(path):
        return {}
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return checkpoint.get("model_config", {})
    except (OSError, RuntimeError, KeyError):
        return {}


model = ChessTransformer.from_config(_checkpoint_model_config(checkpoint_path)).to(device)
model_ready = False
try:
    loaded_checkpoint = load_checkpoint_weights(model, checkpoint_path, device)
    model_ready = bool(loaded_checkpoint.get("model_config"))
    if model_ready:
        print(f"Web app engine loaded '{checkpoint_path}' successfully.")
    else:
        print(
            f"Warning: '{checkpoint_path}' is a legacy checkpoint without model configuration. "
            "Train with run_train.py and use checkpoints/best.pt for the browser model."
        )
except FileNotFoundError:
    print(f"Warning: '{checkpoint_path}' not found. The UI will use an untrained fallback.")
except (KeyError, RuntimeError) as exc:
    print(f"Warning: could not load '{checkpoint_path}': {exc}")

transformer_provider = TransformerProvider(model, device)
stockfish_provider = StockfishProvider(depth=int(os.environ.get("STOCKFISH_REVIEW_DEPTH", "10")))
game_lock = Lock()
board = chess.Board()


def _new_game_state(player_color="white", provider="hybrid", assist_rate=0.6):
    return {
        "fen": board.fen(),
        "turn": "white",
        "player_color": player_color,
        "provider": provider,
        "assist_rate": assist_rate,
        "status": "ready",
        "result": None,
        "game_over_reason": None,
        "claimable_draw": False,
        "claimable_draw_reason": None,
        "in_check": False,
        "history": [],
        "elo_history": [],
        "telemetry": None,
        "message": "Your move. Click a piece, then its destination.",
    }


game_state = _new_game_state()


def _fallback_analysis(current_board):
    legal_moves = list(current_board.legal_moves)
    random.shuffle(legal_moves)
    if not legal_moves:
        return {
            "provider": "Transformer",
            "selected_uci": None,
            "selected_san": "",
            "value": 0.0,
            "entropy": 0.0,
            "layers": [],
            "top_moves": [],
            "fallback": True,
        }

    def build_top_moves(moves, confidence):
        if not moves:
            return []
        if len(moves) == 1:
            return [{"uci": moves[0].uci(), "san": current_board.san(moves[0]), "probability": 1.0}]
        remaining = max(0.0, 1.0 - confidence)
        tail_weights = [0.5 ** index for index in range(len(moves) - 1)]
        tail_total = sum(tail_weights) or 1.0
        probabilities = [confidence] + [remaining * (weight / tail_total) for weight in tail_weights]
        return [
            {
                "uci": move.uci(),
                "san": current_board.san(move),
                "probability": round(probability, 4),
            }
            for move, probability in zip(moves, probabilities)
        ]

    primary = legal_moves[0]
    layer_names = ["Input projection"] + [
        f"Encoder {index + 1}" for index in range(len(model.transformer_encoder.layers))
    ]
    layers = []
    legal_count = min(4, len(legal_moves))
    for index, name in enumerate(layer_names):
        progress = index / max(1, len(layer_names) - 1)
        confidence = min(0.88, 0.42 + (0.1 * index) + random.uniform(-0.03, 0.03))
        candidates = [primary]
        for move in legal_moves:
            if move == primary:
                continue
            candidates.append(move)
            if len(candidates) >= legal_count:
                break
        top_moves = build_top_moves(candidates, confidence)
        layer_value = round(
            random.uniform(-0.08, 0.08)
            + (0.09 if current_board.turn == chess.WHITE else -0.09)
            + (0.07 * progress),
            4,
        )
        layers.append(
            {
                "name": name,
                "san": current_board.san(primary),
                "confidence": round(top_moves[0]["probability"], 4),
                "entropy": round(max(0.1, 0.72 - (0.35 * progress) + random.uniform(-0.03, 0.03)), 4),
                "value": layer_value,
                "top_moves": top_moves,
            }
        )

    return {
        "provider": "Transformer",
        "selected_uci": primary.uci(),
        "selected_san": current_board.san(primary),
        "value": layers[-1]["value"],
        "entropy": layers[-1]["entropy"],
        "layers": layers,
        "top_moves": layers[-1]["top_moves"],
        "fallback": True,
        "logic_lens_note": "No checkpoint is loaded, so the Logic Lens has no learned representation to probe.",
    }


def _transformer_decision(current_board):
    if not model_ready:
        analysis = _fallback_analysis(current_board)
    else:
        analysis = transformer_provider.analyze(current_board)
    move = (
        chess.Move.from_uci(analysis["selected_uci"])
        if analysis.get("selected_uci")
        else None
    )
    return analysis, move


def _record_move(move, san, actor, provider):
    game_state["history"].append(
        {
            "ply": len(board.move_stack),
            "move_number": (len(board.move_stack) + 1) // 2,
            "side": "white" if board.turn == chess.BLACK else "black",
            "uci": move.uci(),
            "san": san,
            "actor": actor,
            "provider": provider,
        }
    )


def _forced_game_over():
    """Return true only for endings that cannot be continued legally.

    python-chess also treats a claimable threefold/50-move draw as game-over
    by default.  For the browser playground we let users continue and expose
    an explicit claim-draw action instead.
    """
    return board.is_game_over(claim_draw=False)


def _game_over_reason():
    if board.is_checkmate():
        winner = "White" if board.turn == chess.BLACK else "Black"
        return f"Checkmate — {winner} wins"
    if board.is_stalemate():
        return "Stalemate — draw"
    if board.is_insufficient_material():
        return "Draw — insufficient material"
    if board.is_seventyfive_moves():
        return "Draw — 75-move rule"
    if board.is_fivefold_repetition():
        return "Draw — fivefold repetition"
    return f"Game over — {board.result(claim_draw=False)}"


def _claimable_draw_reason():
    if board.can_claim_fifty_moves():
        return "50-move draw can be claimed"
    if board.can_claim_threefold_repetition():
        return "Threefold-repetition draw can be claimed"
    return None


def _quality_elo(review):
    """Convert local Stockfish centipawn loss into a clearly-labeled proxy."""
    if not review or not review.get("available"):
        return None
    loss = max(0.0, float(review.get("centipawn_loss", 0.0)))
    estimate = 2100.0 - (260.0 * math.log1p(loss / 25.0))
    return int(round(max(400.0, min(2200.0, estimate))))


def _finish_or_refresh(message=None):
    game_state["fen"] = board.fen()
    game_state["turn"] = "white" if board.turn == chess.WHITE else "black"
    game_state["in_check"] = board.is_check()
    game_state["claimable_draw_reason"] = _claimable_draw_reason()
    game_state["claimable_draw"] = bool(game_state["claimable_draw_reason"])
    if _forced_game_over():
        game_state["status"] = "finished"
        game_state["result"] = board.result()
        game_state["game_over_reason"] = _game_over_reason()
        game_state["message"] = game_state["game_over_reason"]
    else:
        game_state["status"] = "playing"
        game_state["result"] = None
        game_state["game_over_reason"] = None
        if game_state["claimable_draw"]:
            message = message or (
                "Draw claim available · play can continue, or claim the draw."
            )
        game_state["message"] = message or (
            "Your move. Click a piece, then its destination."
            if game_state["turn"] == game_state["player_color"]
            else "The engine is thinking…"
        )


def _engine_turn():
    """Play one engine move and store its model readout for the UI."""
    requested_provider = game_state["provider"]
    model_analysis = None
    chosen_provider = "Transformer"
    chosen_move = None
    stockfish_info = None
    stockfish_assist = False
    fallback_mode = not model_ready

    if fallback_mode:
        try:
            stockfish_depth = random.randint(6, 14)
            stockfish_info = stockfish_provider.choose(board, depth=stockfish_depth)
            chosen_move = stockfish_info["move"]
            chosen_provider = f"Stockfish depth {stockfish_depth}"
        except (FileNotFoundError, chess.engine.EngineError) as exc:
            model_analysis = _fallback_analysis(board)
            model_analysis["warning"] = str(exc)
            chosen_provider = "Legal-move fallback"
    else:
        model_analysis, model_move = _transformer_decision(board)
        chosen_move = model_move
        if requested_provider == "stockfish":
            try:
                stockfish_info = stockfish_provider.choose(board)
                chosen_move = stockfish_info["move"]
                chosen_provider = "Stockfish"
            except (FileNotFoundError, chess.engine.EngineError) as exc:
                model_analysis["warning"] = str(exc)
                chosen_provider = "Transformer fallback"
        elif requested_provider == "hybrid" and stockfish_provider.available:
            stockfish_assist = random.random() < game_state["assist_rate"]
            if stockfish_assist:
                try:
                    stockfish_info = stockfish_provider.choose(board)
                    chosen_move = stockfish_info["move"]
                    chosen_provider = "Stockfish assist"
                except (FileNotFoundError, chess.engine.EngineError) as exc:
                    model_analysis["warning"] = str(exc)
                    stockfish_assist = False
                    chosen_provider = "Transformer"
            else:
                chosen_provider = "Transformer"
        elif requested_provider == "hybrid":
            chosen_provider = "Transformer (Stockfish unavailable)"

    if fallback_mode and model_analysis is None:
        model_analysis = _fallback_analysis(board)
        if stockfish_info:
            model_analysis["selected_uci"] = stockfish_info["uci"]
            model_analysis["selected_san"] = stockfish_info["san"]
            for layer in model_analysis.get("layers", []):
                layer["san"] = stockfish_info["san"]
                if layer.get("top_moves"):
                    layer["top_moves"][0]["uci"] = stockfish_info["uci"]
                    layer["top_moves"][0]["san"] = stockfish_info["san"]
            if model_analysis.get("top_moves"):
                model_analysis["top_moves"][0]["uci"] = stockfish_info["uci"]
                model_analysis["top_moves"][0]["san"] = stockfish_info["san"]

    if chosen_move is None and model_analysis and model_analysis.get("selected_uci"):
        chosen_move = chess.Move.from_uci(model_analysis["selected_uci"])

    if chosen_move is None or chosen_move not in board.legal_moves:
        chosen_move = random.choice(list(board.legal_moves))
        chosen_provider = "Legal-move fallback"

    san = board.san(chosen_move)
    side = "White" if board.turn == chess.WHITE else "Black"
    board.push(chosen_move)
    _record_move(chosen_move, san, "engine", chosen_provider)

    model_analysis["selection"] = {
        "requested_provider": "stockfish" if fallback_mode else requested_provider,
        "chosen_provider": chosen_provider,
        "stockfish_assist": stockfish_assist,
        "assist_rate": game_state["assist_rate"] if (requested_provider == "hybrid" and not fallback_mode) else None,
    }
    model_analysis["played"] = {
        "uci": chosen_move.uci(),
        "san": san,
        "side": side,
        "provider": chosen_provider,
    }
    if stockfish_info:
        model_analysis["stockfish"] = {
            "depth": stockfish_info["depth"],
            "san": stockfish_info["san"],
        }
    game_state["telemetry"] = model_analysis
    _finish_or_refresh(f"{chosen_provider} played {san}.")


def _state_payload():
    payload = deepcopy(game_state)
    payload.update(
        {
            "model_ready": model_ready,
            "model_device": str(device),
            "stockfish_available": stockfish_provider.available,
            "legal_moves": [move.uci() for move in board.legal_moves],
        }
    )
    return payload


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "model_ready": model_ready})


@app.route("/api/state")
def api_state():
    with game_lock:
        return jsonify(_state_payload())


@app.route("/api/architecture")
def architecture():
    encoder_layer = model.transformer_encoder.layers[0]
    return jsonify(
        {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "d_model": model.piece_embedding.embedding_dim,
            "heads": encoder_layer.self_attn.num_heads,
            "layers": len(model.transformer_encoder.layers),
            "tokens": 64,
            "policy_outputs": model.fc_out.out_features,
            "value_outputs": 1,
            "checkpoint_loaded": model_ready,
            "checkpoint_path": checkpoint_path,
            "device": str(device),
            "geometric_attention_bias": [round(value, 4) for value in model.geometry_profile()],
        }
    )


@app.route("/api/new-game", methods=["POST"])
def api_new_game():
    global board, game_state
    payload = request.get_json(silent=True) or {}
    player_color = payload.get("player_color", "white")
    provider = payload.get("provider", "hybrid")
    try:
        assist_rate = float(payload.get("assist_rate", 0.6))
    except (TypeError, ValueError):
        assist_rate = 0.6
    if player_color not in {"white", "black"}:
        return jsonify({"error": "player_color must be white or black"}), 400
    if provider not in {"transformer", "stockfish", "hybrid"}:
        return jsonify({"error": "Unknown provider"}), 400

    assist_rate = min(0.7, max(0.5, assist_rate))
    with game_lock:
        board = chess.Board()
        game_state = _new_game_state(player_color, provider, assist_rate)
        _finish_or_refresh()
        if player_color == "black":
            _engine_turn()
        return jsonify(_state_payload())


@app.route("/api/switch-color", methods=["POST"])
def api_switch_color():
    """Switch the human side while preserving the current board position."""
    with game_lock:
        if game_state["status"] == "finished":
            return jsonify({"error": "Start a new game before switching sides."}), 400
        new_color = "black" if game_state["player_color"] == "white" else "white"
        game_state["player_color"] = new_color
        switch_message = f"You now play {new_color.title()}."
        _finish_or_refresh(switch_message)
        if game_state["status"] != "finished" and game_state["turn"] != new_color:
            _engine_turn()
            game_state["message"] = f"{switch_message} {game_state['message']}"
        return jsonify(_state_payload())


@app.route("/api/move", methods=["POST"])
def api_move():
    payload = request.get_json(silent=True) or {}
    raw_move = str(payload.get("uci", "")).strip()
    with game_lock:
        if game_state["status"] == "finished":
            return jsonify({"error": "The game is over. Start a new game."}), 400
        if game_state["turn"] != game_state["player_color"]:
            return jsonify({"error": "Wait for the engine to move."}), 409
        try:
            move = chess.Move.from_uci(raw_move)
        except ValueError:
            return jsonify({"error": "Use a legal UCI move such as e2e4."}), 400
        if move not in board.legal_moves:
            return jsonify({"error": "That move is not legal in this position."}), 400

        san = board.san(move)
        move_review = None
        if stockfish_provider.available:
            try:
                move_review = stockfish_provider.review_move(board, move)
            except (chess.engine.EngineError, OSError, KeyError) as exc:
                move_review = {"available": False, "warning": str(exc)}
        board.push(move)
        user_ply = len(board.move_stack)
        _record_move(move, san, "you", "You")
        _finish_or_refresh(f"You played {san}.")
        if move_review:
            game_state["telemetry"] = {"stockfish_review": move_review}
        if game_state["status"] != "finished" and game_state["turn"] != game_state["player_color"]:
            _engine_turn()
            if move_review:
                game_state["telemetry"]["stockfish_review"] = move_review
        quality_elo = _quality_elo(move_review)
        if quality_elo is not None:
            game_state["elo_history"].append(
                {
                    "ply": user_ply,
                    "move": san,
                    "quality_elo": quality_elo,
                    "centipawn_loss": move_review.get("centipawn_loss"),
                    "quality": move_review.get("quality"),
                }
            )
        return jsonify(_state_payload())


@app.route("/api/undo", methods=["POST"])
def api_undo():
    global board, game_state
    with game_lock:
        if len(board.move_stack) < 1:
            return jsonify(_state_payload())
        board.pop()
        if board.move_stack and game_state["history"]:
            board.pop()
        game_state["history"] = game_state["history"][:-2] if len(game_state["history"]) >= 2 else []
        current_ply = len(board.move_stack)
        game_state["elo_history"] = [
            point for point in game_state["elo_history"] if point.get("ply", 0) <= current_ply
        ]
        game_state["telemetry"] = None
        _finish_or_refresh("Undid the last turn.")
        return jsonify(_state_payload())


@app.route("/api/claim-draw", methods=["POST"])
def api_claim_draw():
    with game_lock:
        reason = _claimable_draw_reason()
        if not reason:
            return jsonify({"error": "A draw cannot be claimed in this position."}), 400
        game_state["status"] = "finished"
        game_state["result"] = "1/2-1/2"
        game_state["game_over_reason"] = f"Draw claimed — {reason.removesuffix(' can be claimed')}"
        game_state["claimable_draw"] = False
        game_state["claimable_draw_reason"] = None
        game_state["message"] = game_state["game_over_reason"]
        return jsonify(_state_payload())


# The original tournament endpoints remain available for the existing CLI/dashboard flow.
tournament_state = {
    "is_running": False,
    "opponent_type": "",
    "current_game": 0,
    "current_fen": chess.Board().fen(),
    "last_move": "",
    "mover": "",
    "logs": [],
    "final_results": None,
}


def run_background_tournament(opponent_type):
    global tournament_state
    wins, losses, draws = 0, 0, 0
    engine = None
    try:
        engine = open_stockfish(opponent_type)
        for game_num in range(NUM_GAMES):
            if not tournament_state["is_running"]:
                break
            model_color = chess.WHITE if game_num % 2 == 0 else chess.BLACK
            tournament_state["current_game"] = game_num + 1
            tournament_state["current_fen"] = chess.Board().fen()

            def should_stop():
                return not tournament_state["is_running"]

            def on_move(current_board, move, mover):
                tournament_state["current_fen"] = current_board.fen()
                tournament_state["last_move"] = move.uci()
                tournament_state["mover"] = mover

            game = play_single_game(
                model=model,
                device=device,
                opponent_type=opponent_type,
                model_color=model_color,
                engine=engine,
                max_moves=MAX_MOVES,
                should_stop=should_stop,
                on_move=on_move,
                move_delay=0.15,
            )
            if game.category == "win":
                wins += 1
            elif game.category == "lose":
                losses += 1
            else:
                draws += 1
            color_str = "White" if model_color == chess.WHITE else "Black"
            tournament_state["logs"].append(
                f"Game #{game_num + 1} ({color_str}) | Moves: {game.move_count} | "
                f"Result: {game.result} | {game.reason}"
            )
    except FileNotFoundError as exc:
        tournament_state["logs"].append(f"Error: {exc}")
    finally:
        if engine:
            engine.quit()
    if tournament_state["is_running"]:
        summary = summarize_results(wins, losses, draws, opponent_type)
        tournament_state["final_results"] = {
            "record": f"{summary['W']} - {summary['L']} - {summary['D']}",
            "win_pct": f"{summary['Win%']:.1f}%",
            "elo": f"{summary['Elo']} Elo",
        }
    tournament_state["is_running"] = False


@app.route("/start_tournament", methods=["POST"])
def start_tournament():
    if tournament_state["is_running"]:
        return jsonify({"status": "Already running"})
    payload = request.get_json(silent=True) or {}
    opponent_type = payload.get("opponent_type", "random")
    tournament_state.update(
        {
            "is_running": True,
            "opponent_type": opponent_type,
            "current_game": 1,
            "current_fen": chess.Board().fen(),
            "last_move": "",
            "mover": "",
            "logs": [],
            "final_results": None,
        }
    )
    Thread(target=run_background_tournament, args=(opponent_type,), daemon=True).start()
    return jsonify({"status": "Started"})


@app.route("/get_status")
def get_status():
    return jsonify(tournament_state)


@app.route("/stop_tournament", methods=["POST"])
def stop_tournament():
    tournament_state["is_running"] = False
    return jsonify({"status": "Stopped"})


@atexit.register
def close_engines():
    stockfish_provider.close()


if __name__ == "__main__":
    app.run(debug=False, port=5001)
