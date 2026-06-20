import chess
import chess.pgn
import torch
import io

# 1. Map pieces to integer tokens
PIECE_TO_TOKEN = {
    None: 0,
    chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3, chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6
}

def board_to_sequence(board: chess.Board):
    """Converts an 8x8 chess board into a 64-element sequence of tokens."""
    sequence = []
    for square in chess.SQUARES:  # Iterates from A1 to H8
        piece = board.piece_at(square)
        if piece is None:
            sequence.append(0)
        else:
            # White pieces get 1-6, Black pieces get 7-12
            token = PIECE_TO_TOKEN[piece.piece_type]
            if piece.color == chess.BLACK:
                token += 6
            sequence.append(token)
    return torch.tensor(sequence, dtype=torch.long)

def move_to_id(move: chess.Move):
    """Encodes a move into a unique integer ID (0 to 4095)."""
    return move.from_square * 64 + move.to_square

def id_to_move(move_id: int):
    """Decodes a unique integer ID back into a chess.Move."""
    from_square = move_id // 64
    to_square = move_id % 64
    return chess.Move(from_square, to_square)

# --- Quick Test ---
board = chess.Board() # Starting position
print("Initial board sequence:\n", board_to_sequence(board).view(8, 8))

def process_pgn_games(pgn_string):
    pgn_file = io.StringIO(pgn_string)
    X, Y = [], []
    
    while True:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break
        
        board = game.board()
        for move in game.mainline_moves():
            # Store the current board state and the move played from it
            X.append(board_to_sequence(board))
            Y.append(move_to_id(move))
            board.push(move) # Move to next position
            
    return torch.stack(X), torch.tensor(Y, dtype=torch.long)

# Sample PGN data (1 short game)
sample_pgn = "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 1-0"
X_train, Y_train = process_pgn_games(sample_pgn)

print(f"\nProcessed dataset shapes:")
print(f"X (Board States): {X_train.shape}")  # (Num_moves, 64)
print(f"Y (Target Moves): {Y_train.shape}")  # (Num_moves,)