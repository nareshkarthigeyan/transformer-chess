import torch
import chess


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    if not hasattr(optimizer, "step") and hasattr(criterion, "step"):
        optimizer, criterion = criterion, optimizer

    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        if len(batch) == 3:
            boards, targets, fens = batch
        elif len(batch) == 2:
            boards, targets = batch
            fens = None
        else:
            raise ValueError(
                "Expected dataloader batches of (boards, targets) "
                "or (boards, targets, fens)."
            )

        boards = boards.to(device)
        targets = targets.to(device)

        logits = model(boards)

        if fens is not None:
            mask = torch.full_like(logits, float("-inf"))
            for i, fen_str in enumerate(fens):
                board = chess.Board(fen_str)
                for move in board.legal_moves:
                    move_id = move.from_square * 64 + move.to_square
                    mask[i, move_id] = 0.0
            logits = logits + mask

        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
            print(f"   Batch {batch_idx + 1}/{len(dataloader)} | Step Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pt"):
    state = {
        "epoch": epoch,
        "model_state_dict": (
            model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        ),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(state, filename)
    print(f"Checkpoint successfully locked to disk: {filename}")
