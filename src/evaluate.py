# src/evaluate.py
import torch
import torch.nn.functional as F
from .data_loader import board_to_sequence, board_to_state_tensor

def get_best_move(model, board, device):
    """
    Takes a python-chess board, masks out illegal moves, 
    and predicts the highest probability legal move.
    """
    model.eval()
    with torch.no_grad():
        # 1. Convert current board to tensor and push to device
        board_seq = board_to_sequence(board).unsqueeze(0).to(device)
        state_features = board_to_state_tensor(board).unsqueeze(0).to(device)
        
        # 2. Get move probabilities (logits)
        logits = model(board_seq, state_features=state_features)  # Shape: (1, 4096)
        probabilities = F.softmax(logits, dim=-1).squeeze(0)
        
        # 3. Filter for ONLY legal moves (Illegal Move Masking)
        legal_moves = list(board.legal_moves)
        
        # Fallback if no legal moves exist (game over)
        if not legal_moves:
            return None
            
        best_move = None
        best_prob = -1.0
        
        for move in legal_moves:
            # Map move back to its unique ID to check its probability
            move_id = move.from_square * 64 + move.to_square
            move_prob = probabilities[move_id].item()
            
            # The compact source/destination policy has one bucket for all
            # promotion pieces. Prefer queen on an exact tie, which is the
            # conventional default and remains fully legal.
            prefer_queen = (
                move_prob == best_prob
                and move.promotion == chess.QUEEN
                and (best_move is None or best_move.promotion != chess.QUEEN)
            )
            if move_prob > best_prob or prefer_queen:
                best_prob = move_prob
                best_move = move
                
        return best_move
