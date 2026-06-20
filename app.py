# app.py
import torch
import chess
from flask import Flask, render_template, request, jsonify
from src.model import ChessTransformer
from src.evaluate import get_best_move

app = Flask(__name__)

# Load model globally on startup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ChessTransformer().to(device)

try:
    checkpoint = torch.load("checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("🚀 Model successfully loaded for Web GUI!")
except FileNotFoundError:
    print("⚠️ Warning: 'checkpoint.pt' not found. AI will make random legal moves until trained.")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/move", methods=["POST"])
def move():
    data = request.json
    fen = data.get("fen")
    
    board = chess.Board(fen)
    if board.is_game_over():
        return jsonify({"status": "game_over", "result": board.result()})
    
    # Run the board tensor through the transformer architecture
    ai_move = get_best_move(model, board, device)
    
    if ai_move is None:
        # Fallback if no moves are found
        ai_move = list(board.legal_moves)[0]
        
    return jsonify({
        "status": "success",
        "from": chess.square_name(ai_move.from_square),
        "to": chess.square_name(ai_move.to_square),
        "san": board.san(ai_move)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)