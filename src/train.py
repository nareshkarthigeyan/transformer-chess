import torch
import chess


def _apply_legal_move_id_mask(logits, legal_move_ids):
    mask = torch.full_like(logits, float("-inf"))
    valid = legal_move_ids >= 0
    if valid.any():
        rows = torch.arange(logits.size(0), device=logits.device).unsqueeze(1)
        rows = rows.expand_as(legal_move_ids)[valid]
        cols = legal_move_ids[valid]
        mask[rows, cols] = 0.0
    return logits + mask


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    value_criterion=None,
    value_loss_weight=0.25,
):
    if not hasattr(optimizer, "step") and hasattr(criterion, "step"):
        optimizer, criterion = criterion, optimizer

    if value_criterion is None:
        value_criterion = torch.nn.MSELoss()

    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        values = None
        legal_move_ids = None

        if len(batch) == 4:
            boards, targets, values, legal_move_ids = batch
            fens = None
        elif len(batch) == 3:
            boards, targets, fens = batch
        elif len(batch) == 2:
            boards, targets = batch
            fens = None
        else:
            raise ValueError(
                "Expected dataloader batches of (boards, targets), "
                "(boards, targets, fens), or "
                "(boards, targets, values, legal_move_ids)."
            )

        boards = boards.to(device)
        targets = targets.to(device)
        if values is not None:
            values = values.to(device)
        if legal_move_ids is not None:
            legal_move_ids = legal_move_ids.to(device)

        if values is not None:
            logits, predicted_values = model(boards, return_value=True)
        else:
            logits = model(boards)
            predicted_values = None

        if legal_move_ids is not None:
            logits = _apply_legal_move_id_mask(logits, legal_move_ids)
        elif fens is not None:
            mask = torch.full_like(logits, float("-inf"))
            for i, fen_str in enumerate(fens):
                board = chess.Board(fen_str)
                for move in board.legal_moves:
                    move_id = move.from_square * 64 + move.to_square
                    mask[i, move_id] = 0.0
            logits = logits + mask

        policy_loss = criterion(logits, targets)
        if values is not None:
            value_loss = value_criterion(predicted_values, values)
            loss = policy_loss + value_loss_weight * value_loss
        else:
            value_loss = torch.zeros((), device=device)
            loss = policy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
            print(
                f"   Batch {batch_idx + 1}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Policy: {policy_loss.item():.4f} | "
                f"Value: {value_loss.item():.4f}"
            )

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


def load_checkpoint_weights(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing:
        print(f"Checkpoint missing newly added parameters: {', '.join(missing)}")
    if unexpected:
        print(f"Checkpoint had unused parameters: {', '.join(unexpected)}")
    return checkpoint
