import torch
import chess
from threading import Thread
from flask import Flask, render_template, request, jsonify

from src import (
    ChessTransformer,
    NUM_GAMES,
    open_stockfish,
    play_single_game,
    summarize_results,
)

app = Flask(__name__)

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

# Load model weights on startup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ChessTransformer().to(device)
try:
    checkpoint = torch.load("checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Web app engine loaded 'checkpoint.pt' successfully.")
except FileNotFoundError:
    print("Warning: 'checkpoint.pt' not found. Train the model first.")


def run_background_tournament(opponent_type):
    global tournament_state
    wins, losses, draws = 0, 0, 0

    try:
        engine = open_stockfish(opponent_type)
    except FileNotFoundError as exc:
        tournament_state["logs"].append(f"Error: {exc}")
        tournament_state["final_results"] = None
        tournament_state["is_running"] = False
        return

    try:
        for game_num in range(NUM_GAMES):
            if not tournament_state["is_running"]:
                break

            model_color = chess.WHITE if game_num % 2 == 0 else chess.BLACK

            tournament_state["current_game"] = game_num + 1
            tournament_state["current_fen"] = chess.Board().fen()

            def should_stop():
                return not tournament_state["is_running"]

            def on_move(board, move, mover):
                tournament_state["current_fen"] = board.fen()
                tournament_state["last_move"] = move.uci()
                tournament_state["mover"] = mover

            game = play_single_game(
                model=model,
                device=device,
                opponent_type=opponent_type,
                model_color=model_color,
                engine=engine,
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
            log_entry = (
                f"Game #{game_num + 1} ({color_str}) | Moves: {game.move_count} | "
                f"Result: {game.result} | {game.reason}"
            )
            tournament_state["logs"].append(log_entry)
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
    else:
        tournament_state["final_results"] = None

    tournament_state["is_running"] = False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start_tournament", methods=["POST"])
def start_tournament():
    global tournament_state
    if tournament_state["is_running"]:
        return jsonify({"status": "Already running"})

    payload = request.get_json(silent=True) or {}
    opp_type = payload.get("opponent_type", "random")
    tournament_state = {
        "is_running": True,
        "opponent_type": opp_type,
        "current_game": 1,
        "current_fen": chess.Board().fen(),
        "last_move": "",
        "mover": "",
        "logs": [],
        "final_results": None,
    }

    Thread(target=run_background_tournament, args=(opp_type,), daemon=True).start()
    return jsonify({"status": "Started"})


@app.route("/get_status", methods=["GET"])
def get_status():
    return jsonify(tournament_state)


@app.route("/stop_tournament", methods=["POST"])
def stop_tournament():
    global tournament_state
    tournament_state["is_running"] = False
    return jsonify({"status": "Stopped"})


if __name__ == "__main__":
    app.run(debug=False, port=5001)
