"""Training, metrics, and checkpoint compatibility helpers."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, Optional

import chess
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


def _apply_legal_move_id_mask(logits: torch.Tensor, legal_move_ids: torch.Tensor) -> torch.Tensor:
    """Mask every policy bucket that is not legal in its own board position."""
    mask = torch.full_like(logits, float("-inf"))
    valid = legal_move_ids >= 0
    if valid.any():
        rows = torch.arange(logits.size(0), device=logits.device).unsqueeze(1).expand_as(legal_move_ids)
        mask[rows[valid], legal_move_ids[valid]] = 0.0
    return logits + mask


def _unpack_batch(batch):
    """Support the new distilled cache and old sanity/PGN tuple layouts."""
    if isinstance(batch, dict):
        return (
            batch["boards"], batch.get("state_features"), batch["targets"], batch.get("values"),
            batch.get("legal_move_ids"), batch.get("teacher_top_move_ids"), batch.get("teacher_top_probs"), None,
        )
    if len(batch) == 7:
        return (*batch, None)
    if len(batch) == 5:
        # Legacy ChessPGNDataset: boards, state, targets, fens, unused.
        boards, states, targets, fens, unused = batch
        return boards, states, targets, None, None, None, None, fens
    if len(batch) == 4:
        boards, second, third, fourth = batch
        # Newer legacy PGN dataset: boards, state_features, targets, fens.
        if isinstance(second, torch.Tensor) and second.ndim == 2 and second.size(1) == 5:
            return boards, second, third, None, None, None, None, fourth
        return boards, None, second, third, fourth, None, None, None
    if len(batch) == 3:
        boards, targets, fens = batch
        return boards, None, targets, None, None, None, None, fens
    if len(batch) == 2:
        boards, targets = batch
        return boards, None, targets, None, None, None, None, None
    raise ValueError("Unsupported dataloader batch layout.")


def _mask_from_fens(logits: torch.Tensor, fens) -> torch.Tensor:
    legal_ids = torch.full((len(fens), 256), -1, dtype=torch.long, device=logits.device)
    for index, fen in enumerate(fens):
        board = chess.Board(fen)
        for move_index, move in enumerate(board.legal_moves):
            legal_ids[index, move_index] = move.from_square * 64 + move.to_square
    return _apply_legal_move_id_mask(logits, legal_ids)


def _teacher_policy_loss_per_example(
    logits: torch.Tensor,
    teacher_top_move_ids: Optional[torch.Tensor],
    teacher_top_probs: Optional[torch.Tensor],
) -> torch.Tensor:
    if teacher_top_move_ids is None or teacher_top_probs is None:
        return torch.zeros(logits.size(0), device=logits.device)
    valid = teacher_top_move_ids >= 0
    if not valid.any():
        return torch.zeros(logits.size(0), device=logits.device)
    probabilities = teacher_top_probs * valid.to(teacher_top_probs.dtype)
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)
    selected_log_probs = F.log_softmax(logits, dim=1).gather(
        1, teacher_top_move_ids.clamp_min(0)
    )
    # Invalid padded slots can point at an illegal bucket (log p = -inf).
    # Replace them before multiplication: 0 * -inf is NaN in IEEE arithmetic.
    selected_log_probs = torch.where(valid, selected_log_probs, torch.zeros_like(selected_log_probs))
    return -(probabilities * selected_log_probs).sum(dim=1)


def _teacher_policy_loss(
    logits: torch.Tensor,
    teacher_top_move_ids: Optional[torch.Tensor],
    teacher_top_probs: Optional[torch.Tensor],
) -> torch.Tensor:
    return _teacher_policy_loss_per_example(
        logits, teacher_top_move_ids, teacher_top_probs
    ).mean()


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def run_epoch(
    model,
    dataloader,
    device: torch.device,
    optimizer=None,
    value_loss_weight: float = 0.25,
    teacher_policy_weight: float = 0.35,
    scaler=None,
    amp: bool = False,
    log_every: int = 25,
    log_fn: Optional[Callable[[str], None]] = None,
):
    """Run one train or validation epoch and return presentation-ready metrics."""
    training = optimizer is not None
    model.train(training)
    metric_totals = {"loss": 0.0, "policy_loss": 0.0, "teacher_policy_loss": 0.0, "value_loss": 0.0, "top1": 0.0, "value_mae": 0.0}
    examples = 0

    for batch_index, batch in enumerate(dataloader, start=1):
        boards, states, targets, values, legal_move_ids, teacher_ids, teacher_probs, fens = _unpack_batch(batch)
        boards = boards.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        states = states.to(device, non_blocking=True) if states is not None else None
        values = values.to(device, non_blocking=True) if values is not None else None
        legal_move_ids = legal_move_ids.to(device, non_blocking=True) if legal_move_ids is not None else None
        teacher_ids = teacher_ids.to(device, non_blocking=True) if teacher_ids is not None else None
        teacher_probs = teacher_probs.to(device, non_blocking=True) if teacher_probs is not None else None

        with torch.set_grad_enabled(training), _autocast_context(device, amp):
            logits, predicted_values = model(boards, state_features=states, return_value=True)
            if legal_move_ids is not None:
                logits = _apply_legal_move_id_mask(logits, legal_move_ids)
            elif fens is not None:
                logits = _mask_from_fens(logits, fens)

            hard_policy_loss = F.cross_entropy(logits, targets)
            soft_policy_loss = _teacher_policy_loss(logits, teacher_ids, teacher_probs)
            policy_loss = (1 - teacher_policy_weight) * hard_policy_loss + teacher_policy_weight * soft_policy_loss
            value_loss = F.mse_loss(predicted_values, values) if values is not None else torch.zeros((), device=device)
            loss = policy_loss + value_loss_weight * value_loss

        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = boards.size(0)
        examples += batch_size
        metric_totals["loss"] += float(loss.detach()) * batch_size
        metric_totals["policy_loss"] += float(hard_policy_loss.detach()) * batch_size
        metric_totals["teacher_policy_loss"] += float(soft_policy_loss.detach()) * batch_size
        metric_totals["value_loss"] += float(value_loss.detach()) * batch_size
        metric_totals["top1"] += float((logits.argmax(dim=1) == targets).sum().detach())
        if values is not None:
            metric_totals["value_mae"] += float((predicted_values - values).abs().sum().detach())

        if log_fn and (batch_index % log_every == 0 or batch_index == len(dataloader)):
            log_fn(
                f"{'train' if training else 'valid'} batch {batch_index}/{len(dataloader)} "
                f"loss={float(loss.detach()):.4f} hard_policy={float(hard_policy_loss.detach()):.4f} "
                f"soft_policy={float(soft_policy_loss.detach()):.4f} value={float(value_loss.detach()):.4f}"
            )

    if examples == 0:
        raise RuntimeError("Dataloader produced no batches.")
    return {key: value / examples for key, value in metric_totals.items()}


@torch.no_grad()
def collect_hard_example_weights(
    model,
    dataset,
    indices,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 0,
    value_loss_weight: float = 0.25,
    teacher_policy_weight: float = 0.35,
    amp: bool = False,
):
    """Score training rows by current error for the next curriculum round.

    The returned array is aligned with ``indices`` and has mean one. A later
    round can multiply it by the tactical/endgame prior, so difficult examples
    are revisited without discarding ordinary positions.
    """
    if len(indices) == 0:
        return np.zeros(0, dtype=np.float32)
    loader = DataLoader(
        Subset(dataset, list(indices)), batch_size=batch_size, shuffle=False,
        num_workers=max(0, num_workers), pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    was_training = model.training
    model.eval()
    losses = []
    for batch in loader:
        boards, states, targets, values, legal_move_ids, teacher_ids, teacher_probs, fens = _unpack_batch(batch)
        boards = boards.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        states = states.to(device, non_blocking=True) if states is not None else None
        values = values.to(device, non_blocking=True) if values is not None else None
        legal_move_ids = legal_move_ids.to(device, non_blocking=True) if legal_move_ids is not None else None
        teacher_ids = teacher_ids.to(device, non_blocking=True) if teacher_ids is not None else None
        teacher_probs = teacher_probs.to(device, non_blocking=True) if teacher_probs is not None else None
        with _autocast_context(device, amp):
            logits, predicted_values = model(boards, state_features=states, return_value=True)
            if legal_move_ids is not None:
                logits = _apply_legal_move_id_mask(logits, legal_move_ids)
            elif fens is not None:
                logits = _mask_from_fens(logits, fens)
            hard = F.cross_entropy(logits, targets, reduction="none")
            soft = _teacher_policy_loss_per_example(logits, teacher_ids, teacher_probs)
            value = ((predicted_values - values) ** 2) if values is not None else torch.zeros_like(hard)
            losses.append(((1 - teacher_policy_weight) * hard + teacher_policy_weight * soft + value_loss_weight * value).float().cpu())
    if was_training:
        model.train()
    scores = torch.cat(losses).numpy().astype(np.float32, copy=False)
    mean = float(scores.mean()) if scores.size else 1.0
    return (1.0 + scores / max(mean, 1e-6)).astype(np.float32, copy=False)


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion=None,
    device=None,
    value_criterion=None,
    value_loss_weight=0.25,
):
    """Backward-compatible wrapper retained for the original sanity script."""
    if device is None and isinstance(criterion, torch.device):
        device, criterion = criterion, None
    if device is None:
        raise ValueError("A torch device is required.")
    metrics = run_epoch(
        model, dataloader, device=device, optimizer=optimizer,
        value_loss_weight=value_loss_weight, teacher_policy_weight=0.0,
    )
    return metrics["loss"]


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    loss: float,
    filename: str = "checkpoint.pt",
    scheduler=None,
    scaler=None,
    metrics: Optional[dict] = None,
    global_step: int = 0,
    run_config: Optional[dict] = None,
    curriculum_state: Optional[dict] = None,
):
    """Save a full, resumable checkpoint including model configuration."""
    state = {
        "format_version": 2,
        "epoch": epoch,
        "global_step": global_step,
        "model_config": getattr(model.module if hasattr(model, "module") else model, "config", {}),
        "model_state_dict": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "loss": loss,
        "metrics": metrics or {},
        "run_config": run_config or {},
        "curriculum_state": curriculum_state or {},
    }
    temporary = f"{filename}.tmp"
    torch.save(state, temporary)
    import os
    os.replace(temporary, filename)


def load_checkpoint_weights(model, checkpoint_path, device, optimizer=None, scheduler=None, scaler=None):
    """Load compatible weights and optionally full optimiser state.

    Shape-mismatched keys are ignored with an explicit message. This makes the
    old 4,096-output checkpoint usable as a starting point when architectural
    additions are made, instead of crashing behind ``strict=False``.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source = checkpoint.get("model_state_dict", checkpoint)
    target = model.state_dict()
    compatible = {key: value for key, value in source.items() if key in target and target[key].shape == value.shape}
    skipped = sorted(set(source) - set(compatible))
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if skipped:
        print(f"Checkpoint skipped {len(skipped)} incompatible/new parameter tensors.")
    if missing:
        print(f"Checkpoint missing {len(missing)} parameter tensors; they keep fresh initialization.")
    if unexpected:
        print(f"Checkpoint had {len(unexpected)} unused parameter tensors.")
    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except (ValueError, RuntimeError) as exc:
            print(f"Optimizer state was not compatible and was skipped: {exc}")
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and scaler.is_enabled() and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint
