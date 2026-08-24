import atexit
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
model = ChessTransformer().to(device)
model_ready = False
try:
    load_checkpoint_weights(model, "checkpoint.pt", device)
    model_ready = True
    print("Web app engine loaded 'checkpoint.pt' successfully.")
except FileNotFoundError:
    print("Warning: 'checkpoint.pt' not found. The UI will use an untrained fallback.")
except (KeyError, RuntimeError) as exc:
    print(f"Warning: could not load 'checkpoint.pt': {exc}")

transformer_provider = TransformerProvider(model, device)
stockfish_provider = StockfishProvider(depth=10)
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
        "history": [],
        "telemetry": None,
        "message": "Your move. Click a piece, then its destination.",
    }


game_state = _new_game_state()


def _fallback_analysis(current_board):
    legal_moves = list(current_board.legal_moves)
    random.shuffle(legal_moves)
    top_moves = [
        {
            "uci": move.uci(),
            "san": current_board.san(move),
            "probability": round(1 / max(1, len(legal_moves)), 4),
        }
        for move in legal_moves[:4]
    ]
    return {
        "provider": "Transformer",
        "selected_uci": legal_moves[0].uci() if legal_moves else None,
        "selected_san": current_board.san(legal_moves[0]) if legal_moves else "",
        "value": 0.0,
        "entropy": 1.0,
        "layers": [],
        "top_moves": top_moves,
        "fallback": True,
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


def _finish_or_refresh(message=None):
    game_state["fen"] = board.fen()
    game_state["turn"] = "white" if board.turn == chess.WHITE else "black"
    if board.is_game_over():
        game_state["status"] = "finished"
        game_state["result"] = board.result()
        game_state["message"] = message or f"Game over · {board.result()}"
    else:
        game_state["status"] = "playing"
        game_state["message"] = message or (
            "Your move. Click a piece, then its destination."
            if game_state["turn"] == game_state["player_color"]
            else "The engine is thinking…"
        )


def _engine_turn():
    """Play one engine move and store its model readout for the UI."""
    model_analysis, model_move = _transformer_decision(board)
    requested_provider = game_state["provider"]
    chosen_provider = "Transformer"
    chosen_move = model_move
    stockfish_info = None
    stockfish_assist = False

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

    if chosen_move is None or chosen_move not in board.legal_moves:
        chosen_move = random.choice(list(board.legal_moves))
        chosen_provider = "Legal-move fallback"

    san = board.san(chosen_move)
    side = "White" if board.turn == chess.WHITE else "Black"
    board.push(chosen_move)
    _record_move(chosen_move, san, "engine", chosen_provider)

    model_analysis["selection"] = {
        "requested_provider": requested_provider,
        "chosen_provider": chosen_provider,
        "stockfish_assist": stockfish_assist,
        "assist_rate": game_state["assist_rate"] if requested_provider == "hybrid" else None,
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
            "device": str(device),
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
        if player_color == "black":
            _engine_turn()
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
        board.push(move)
        _record_move(move, san, "you", "You")
        _finish_or_refresh(f"You played {san}.")
        if not board.is_game_over() and game_state["turn"] != game_state["player_color"]:
            _engine_turn()
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
        game_state["telemetry"] = None
        _finish_or_refresh("Undid the last turn.")
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
