import io
import chess
import chess.pgn
import torch

PIECE_TO_TOKEN = {
    None: 0,
    chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3, 
    chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6
}

def board_to_sequence(board: chess.Board) -> torch.Tensor:
    """Converts an 8x8 chess board into a 64-element sequence of tokens."""
    sequence = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            sequence.append(0)
        else:
            token = PIECE_TO_TOKEN[piece.piece_type]
            if piece.color == chess.BLACK:
                token += 6
            sequence.append(token)
    return torch.tensor(sequence, dtype=torch.long)

def move_to_id(move: chess.Move) -> int:
    """Encodes a move into a unique integer ID (0 to 4095)."""
    return move.from_square * 64 + move.to_square

def id_to_move(move_id: int) -> chess.Move:
    """Decodes a unique integer ID back into a chess.Move."""
    from_square = move_id // 64
    to_square = move_id % 64
    return chess.Move(from_square, to_square)

def process_pgn_games(pgn_string: str):
    """Parses a PGN string into features (X) and labels (Y)."""
    pgn_file = io.StringIO(pgn_string)
    X, Y = [], []
    
    while True:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break
        
        board = game.board()
        for move in game.mainline_moves():
            X.append(board_to_sequence(board))
            Y.append(move_to_id(move))
            board.push(move)
            
    return torch.stack(X), torch.tensor(Y, dtype=torch.long)