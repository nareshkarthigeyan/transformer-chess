# play.py
import torch
import chess
from src.model import ChessTransformer
from src.evaluate import get_best_move

def play_against_ai():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading AI using device: {device}")
    
    # 1. Initialize and load model weights from checkpoint
    model = ChessTransformer().to(device)
    try:
        checkpoint = torch.load("checkpoint.pt", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Successfully loaded trained 'checkpoint.pt'!")
    except FileNotFoundError:
        print("Error: 'checkpoint.pt' not found. Please run training first!")
        return

    # 2. Setup the chess board loop
    board = chess.Board()
    print("\nWelcome to Chess vs. Transformer AI.\n")

    while not board.is_game_over():
        print("\n--- Current Board State ---")
        print(board)
        print("---------------------------")
        
        if board.turn == chess.WHITE:
            # Human Player (White)
            human_move = input("\nYour Move (UCI format, e.g., e2e4, g1f3): ").strip()
            if human_move.lower() == 'quit':
                print("Exiting game.")
                break
            try:
                move = chess.Move.from_uci(human_move)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move for this position! Try again.")
            except ValueError:
                print("Invalid input format. Use Standard algebraic UCI notation (like 'd2d4').")
        else:
            # AI Player (Black)
            print("\nAI is processing board...")
            ai_move = get_best_move(model, board, device)
            
            if ai_move is None:
                print("AI has no legal moves left.")
                break
                
            print(f"AI plays: {ai_move}")
            board.push(ai_move)
            
    print("\nGame over.")
    print("Final Result:", board.result())

if __name__ == "__main__":
    play_against_ai()
