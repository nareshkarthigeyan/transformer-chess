"""Single-command, resumable curriculum training for the chess transformer."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from src.data_loader import (
    DATASET_VERSION,
    DEFAULT_DATASET_PATH,
    ChessNumpyDataset,
    build_stockfish_distilled_dataset,
)
from src.model import ChessTransformer
from src.train import (
    collect_hard_example_weights,
    load_checkpoint_weights,
    run_epoch,
    save_checkpoint,
)


# ``strong`` is the default used by the cloud launcher. The other presets are
# useful for a fast smoke test or for reproducing a smaller presentation run.
PRESETS = {
    "smoke": {
        "max_positions": 1_000, "stockfish_time": 0.01,
        "d_model": 96, "nhead": 4, "num_layers": 3, "dim_feedforward": 384,
        "batch_size": 128, "sample_every": 2, "curriculum": False,
        "human_epochs": 0, "distill_epochs": 3, "hard_epochs": 0, "rounds": 1,
        "label_workers": 1, "label_batch_size": 128,
    },
    "presentation": {
        "max_positions": 60_000, "stockfish_time": 0.08,
        "d_model": 128, "nhead": 4, "num_layers": 4, "dim_feedforward": 512,
        "batch_size": 256, "sample_every": 1, "curriculum": True,
        "human_epochs": 1, "distill_epochs": 6, "hard_epochs": 1, "rounds": 1,
        "label_workers": 1, "label_batch_size": 256,
    },
    "strong": {
        "max_positions": 200_000, "stockfish_time": 0.08,
        "d_model": 128, "nhead": 4, "num_layers": 4, "dim_feedforward": 512,
        "batch_size": 256, "sample_every": 1, "curriculum": True,
        "human_epochs": 3, "distill_epochs": 10, "hard_epochs": 2, "rounds": 2,
        "label_workers": 2, "label_batch_size": 256,
    },
    "research": {
        "max_positions": 300_000, "stockfish_time": None, "stockfish_depth": 15,
        "d_model": 192, "nhead": 6, "num_layers": 6, "dim_feedforward": 768,
        "batch_size": 192, "sample_every": 1, "curriculum": True,
        "human_epochs": 3, "distill_epochs": 12, "hard_epochs": 3, "rounds": 2,
        "label_workers": 2, "label_batch_size": 256,
    },
}


class RunLogger:
    """Plain-text + JSONL audit trail suitable for a project report."""

    def __init__(self, text_path: str, metrics_path: str):
        self.text_path = Path(text_path)
        self.metrics_path = Path(metrics_path)
        self.text_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def write(self, message: str) -> None:
        line = f"[{self._timestamp()}] {message}"
        print(line, flush=True)
        with self.text_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def metrics(self, payload: dict) -> None:
        record = {"timestamp": self._timestamp(), **payload}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a geometry-biased chess transformer.")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="presentation")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--no-resume-dataset", action="store_true")
    parser.add_argument("--max-games-per-file", type=int, default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--sample-every", type=int, default=None)
    parser.add_argument("--min-ply", type=int, default=0)
    parser.add_argument("--max-ply", type=int, default=None)
    parser.add_argument("--stockfish-path", default=os.environ.get("STOCKFISH_PATH"))
    parser.add_argument("--stockfish-depth", type=int, default=None)
    parser.add_argument("--stockfish-time", type=float, default=None)
    parser.add_argument("--teacher-multipv", type=int, default=4)
    parser.add_argument("--policy-temperature", type=int, default=120)
    parser.add_argument("--stockfish-threads", type=int, default=2)
    parser.add_argument("--stockfish-hash-mb", type=int, default=256)
    parser.add_argument("--label-workers", type=int, default=None, help="Parallel Stockfish worker processes (1 disables parallel labelling).")
    parser.add_argument("--label-batch-size", type=int, default=None, help="Positions queued per parallel labelling batch.")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dim-feedforward", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override with one distillation stage.")
    parser.add_argument("--human-epochs", type=int, default=None)
    parser.add_argument("--distill-epochs", type=int, default=None)
    parser.add_argument("--hard-epochs", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tactical-oversample", type=float, default=1.0)
    parser.add_argument("--endgame-oversample", type=float, default=1.0)
    parser.add_argument("--opening-oversample", type=float, default=0.5)
    parser.add_argument("--hard-oversample", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--value-loss-weight", type=float, default=0.25)
    parser.add_argument("--teacher-policy-weight", type=float, default=0.35)
    parser.add_argument("--val-fraction", type=float, default=0.08)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--accelerator", choices=("auto", "cuda", "mps", "cpu", "xla"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", default=None, help="Full checkpoint path, or 'auto' for checkpoints/last.pt.")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--save-every", type=int, default=4, help="Keep a numbered full checkpoint every N epochs.")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--log-path", default="logs/training.log.txt")
    parser.add_argument("--metrics-path", default="logs/training_metrics.jsonl")
    parser.add_argument("--train-only", action="store_true", help="Fail if the cache is absent or stale instead of building it.")
    return parser.parse_args(argv)


def apply_preset(args):
    defaults = PRESETS[args.preset]
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    if args.stockfish_depth is None:
        args.stockfish_depth = 15
    if args.preset != "research" and args.stockfish_time is None:
        args.stockfish_time = defaults.get("stockfish_time")
    return args


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA GPU is available.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        return torch.device("mps")
    if requested == "xla":
        try:
            import torch_xla.core.xla_model as xm
        except ImportError as exc:
            raise RuntimeError(
                "TPU/XLA requires torch-xla. Use a GPU runtime for this one-command pipeline, "
                "or install a Colab-compatible torch-xla build first."
            ) from exc
        return xm.xla_device()
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def _split_dataset(dataset, val_fraction: float, seed: int):
    """Split by complete games so positions from one game never leak to validation."""
    if len(dataset) < 10:
        return dataset, None
    game_ids = np.asarray(dataset.game_ids).astype(str)
    unique_games = np.unique(game_ids)
    if unique_games.size < 2:
        return dataset, None
    rng = np.random.default_rng(seed)
    shuffled = unique_games[rng.permutation(unique_games.size)]
    validation_games = max(1, int(round(unique_games.size * val_fraction)))
    validation_games = min(validation_games, unique_games.size - 1)
    val_set = set(shuffled[:validation_games].tolist())
    val_mask = np.asarray([game_id in val_set for game_id in game_ids])
    train_indices = np.flatnonzero(~val_mask).tolist()
    validation_indices = np.flatnonzero(val_mask).tolist()
    return Subset(dataset, train_indices), Subset(dataset, validation_indices)


def _subset_indices(subset):
    if isinstance(subset, Subset):
        return np.asarray(subset.indices, dtype=np.int64)
    return np.arange(len(subset), dtype=np.int64)


def _curriculum_weights(dataset, indices, stage_name, hard_scores, args):
    base = np.asarray(dataset.sampling_weights, dtype=np.float32)[indices].copy()
    if stage_name == "human_pretrain":
        return np.ones(len(indices), dtype=np.float32)
    flags = np.asarray(dataset.priority_flags, dtype=np.float32)[indices]
    if flags.ndim == 2 and flags.shape[1] >= 2:
        base *= 1.0 + args.tactical_oversample * flags[:, 0]
        base *= 1.0 + args.endgame_oversample * flags[:, 1]
    # ECO-aware inverse-frequency prior prevents a few popular openings from
    # dominating a multi-player corpus while keeping every row available.
    if args.opening_oversample > 0:
        codes = np.asarray(dataset.eco_codes)[indices].astype(str)
        frequency = {code: count for code, count in zip(*np.unique(codes, return_counts=True))}
        inverse = np.asarray([1.0 / np.sqrt(frequency[code]) for code in codes], dtype=np.float32)
        inverse /= max(float(inverse.mean()), 1e-6)
        base *= 1.0 + args.opening_oversample * inverse
    if hard_scores is not None:
        hard_scores = np.asarray(hard_scores, dtype=np.float32)
        hard_scores = hard_scores / max(float(hard_scores.mean()), 1e-6)
        base *= 1.0 + args.hard_oversample * np.maximum(hard_scores - 1.0, 0.0)
    return np.maximum(base, 1e-3).astype(np.float32)


def _make_loader(dataset, batch_size, shuffle, num_workers, device, weights=None):
    sampler = None
    if weights is not None:
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double), num_samples=len(dataset), replacement=True
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=shuffle and len(dataset) >= batch_size,
        num_workers=max(0, num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def _checkpoint_model_config(path: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint.get("model_config", {})


def _cache_needs_rebuild(path: str) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            if "metadata" not in data:
                return True
            metadata = json.loads(str(data["metadata"].item()))
            required = {"priority_flags", "sampling_weights", "game_ids", "eco_codes"}
            return int(metadata.get("dataset_version", 0)) < DATASET_VERSION or not required.issubset(data.files)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return True


def _stage_plan(args):
    if args.epochs is not None:
        return [{"name": "distill_override", "target_mode": "teacher", "sampler": "priority", "epochs": args.epochs}]
    if not args.curriculum:
        return [{"name": "distill", "target_mode": "teacher", "sampler": "uniform", "epochs": args.distill_epochs}]
    stages = []
    if args.human_epochs > 0:
        stages.append({"name": "human_pretrain", "target_mode": "human", "sampler": "uniform", "epochs": args.human_epochs})
    for round_number in range(1, args.rounds + 1):
        stages.append({
            "name": f"distill_round_{round_number}", "target_mode": "teacher",
            "sampler": "priority", "epochs": args.distill_epochs,
        })
        if args.hard_epochs > 0:
            stages.append({
                "name": f"hard_examples_round_{round_number}", "target_mode": "teacher",
                "sampler": "hard", "epochs": args.hard_epochs,
            })
    return stages


def main(argv=None):
    args = apply_preset(parse_args(argv))
    logger = RunLogger(args.log_path, args.metrics_path)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = select_device(args.accelerator)
    logger.write(f"run started preset={args.preset} device={device} torch={torch.__version__}")
    logger.write("configuration=" + json.dumps(vars(args), sort_keys=True, default=str))
    if device.type == "cuda":
        logger.write(f"gpu={torch.cuda.get_device_name(device)}")

    if args.rebuild_dataset or not os.path.isfile(args.dataset_path) or _cache_needs_rebuild(args.dataset_path):
        if args.train_only:
            raise FileNotFoundError(f"Dataset cache is absent or older than schema v{DATASET_VERSION}: {args.dataset_path}")
        logger.write("building/resuming Stockfish-distilled dataset")
        result_path = build_stockfish_distilled_dataset(
            data_dir=args.data_dir, output_path=args.dataset_path,
            max_games_per_file=args.max_games_per_file, stockfish_path=args.stockfish_path,
            stockfish_depth=args.stockfish_depth, stockfish_time=args.stockfish_time,
            max_positions=args.max_positions, sample_every=args.sample_every,
            min_ply=args.min_ply, max_ply=args.max_ply, teacher_multipv=args.teacher_multipv,
            policy_temperature=args.policy_temperature, stockfish_threads=args.stockfish_threads,
            stockfish_hash_mb=args.stockfish_hash_mb, label_workers=args.label_workers,
            label_batch_size=args.label_batch_size, resume=not args.no_resume_dataset,
        )
        if result_path != args.dataset_path:
            logger.write(f"dataset labelling paused; resume with the same command ({result_path})")
            return 2

    dataset = ChessNumpyDataset(args.dataset_path)
    metadata = dataset.metadata
    logger.write(
        f"dataset loaded positions={len(dataset):,} games={metadata.get('total_games', len(np.unique(dataset.game_ids))):,} "
        f"tactical={metadata.get('tactical_positions', 'n/a')} endgame={metadata.get('endgame_positions', 'n/a')} "
        f"metadata={json.dumps(metadata, sort_keys=True)}"
    )
    train_set, validation_set = _split_dataset(dataset, args.val_fraction, args.seed)
    train_indices = _subset_indices(train_set)
    logger.write(
        f"game-level split train={len(train_set):,} validation={len(validation_set) if validation_set else 0:,} "
        f"train_games={len(np.unique(dataset.game_ids[train_indices])):,}"
    )

    stages = _stage_plan(args)
    total_epochs = sum(int(stage["epochs"]) for stage in stages)
    if total_epochs < 1:
        raise ValueError("The curriculum has no training epochs.")
    logger.write("curriculum=" + json.dumps(stages, sort_keys=True))

    resume_path = os.path.join(args.checkpoint_dir, "last.pt") if args.resume == "auto" else args.resume
    model_config = _checkpoint_model_config(resume_path) if resume_path and os.path.isfile(resume_path) else {
        "d_model": args.d_model, "nhead": args.nhead, "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward, "dropout": args.dropout,
    }
    model = ChessTransformer.from_config(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    start_epoch = 0
    best_validation_loss = float("inf")
    if resume_path:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Requested resume checkpoint not found: {resume_path}")
        checkpoint = load_checkpoint_weights(model, resume_path, device, optimizer, scheduler, scaler)
        start_epoch = int(checkpoint.get("epoch", 0))
        best_path = os.path.join(args.checkpoint_dir, "best.pt")
        if os.path.isfile(best_path):
            best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
            best_validation_loss = float(best_checkpoint.get("loss", float("inf")))
        else:
            best_validation_loss = float(checkpoint.get("metrics", {}).get("validation", {}).get("loss", float("inf")))
        logger.write(f"resumed checkpoint={resume_path} completed_epochs={start_epoch}/{total_epochs}")

    logger.write(
        f"model parameters={sum(parameter.numel() for parameter in model.parameters()):,} "
        f"geometry_bias={','.join(f'{value:.3f}' for value in model.geometry_profile())}"
    )
    run_config = {**vars(args), "curriculum_stages": stages, "total_epochs": total_epochs}
    completed_epoch = 0
    for stage_index, stage in enumerate(stages):
        dataset.target_mode = stage["target_mode"]
        hard_scores = None
        if stage["sampler"] == "hard":
            round_number = stage["name"].rsplit("_", 1)[-1]
            hard_path = Path(args.checkpoint_dir) / f"hard_weights_round_{round_number}.npy"
            if hard_path.is_file():
                hard_scores = np.load(hard_path)
                if len(hard_scores) != len(train_indices):
                    logger.write(f"discarding stale hard-example weights at {hard_path} (length changed)")
                    hard_scores = None
                else:
                    logger.write(f"loaded hard-example weights from {hard_path}")
            if hard_scores is None:
                logger.write(f"scoring hard examples before {stage['name']}")
                hard_scores = collect_hard_example_weights(
                    model, dataset, train_indices, device, batch_size=args.batch_size,
                    num_workers=args.num_workers, value_loss_weight=args.value_loss_weight,
                    teacher_policy_weight=args.teacher_policy_weight, amp=args.amp,
                )
                np.save(hard_path, hard_scores)
                logger.write(f"saved hard-example weights to {hard_path}")
        weights = None
        if stage["sampler"] in {"priority", "hard"}:
            weights = _curriculum_weights(dataset, train_indices, stage["name"], hard_scores, args)
        train_loader = _make_loader(train_set, args.batch_size, weights is None, args.num_workers, device, weights)
        validation_loader = (
            _make_loader(validation_set, args.batch_size, False, args.num_workers, device)
            if validation_set else None
        )
        logger.write(f"stage={stage['name']} target={stage['target_mode']} sampler={stage['sampler']} epochs={stage['epochs']}")
        for stage_epoch in range(1, int(stage["epochs"]) + 1):
            completed_epoch += 1
            if completed_epoch <= start_epoch:
                continue
            teacher_weight = args.teacher_policy_weight if stage["target_mode"] == "teacher" else 0.0
            train_metrics = run_epoch(
                model, train_loader, device=device, optimizer=optimizer,
                value_loss_weight=args.value_loss_weight, teacher_policy_weight=teacher_weight,
                scaler=scaler, amp=args.amp, log_every=args.log_every, log_fn=logger.write,
            )
            validation_metrics = None
            if validation_loader:
                validation_metrics = run_epoch(
                    model, validation_loader, device=device,
                    value_loss_weight=args.value_loss_weight, teacher_policy_weight=teacher_weight,
                    amp=args.amp, log_every=args.log_every, log_fn=logger.write,
                )
            monitor = (validation_metrics or train_metrics)["loss"]
            scheduler.step()
            metrics = {
                "epoch": completed_epoch, "stage": stage["name"], "stage_epoch": stage_epoch,
                "train": train_metrics, "validation": validation_metrics,
                "learning_rate": optimizer.param_groups[0]["lr"], "monitor_loss": monitor,
            }
            logger.metrics(metrics)
            logger.write(
                f"epoch={completed_epoch}/{total_epochs} stage={stage['name']} train_loss={train_metrics['loss']:.4f} "
                f"train_top1={train_metrics['top1']:.3f} "
                + (f"val_loss={validation_metrics['loss']:.4f} val_top1={validation_metrics['top1']:.3f}" if validation_metrics else "")
            )
            curriculum_state = {"stage_index": stage_index, "stage": stage["name"], "stage_epoch": stage_epoch, "total_epochs": total_epochs}
            last_path = os.path.join(args.checkpoint_dir, "last.pt")
            save_checkpoint(model, optimizer, completed_epoch, monitor, last_path, scheduler, scaler, metrics, run_config=run_config, curriculum_state=curriculum_state)
            if completed_epoch % args.save_every == 0:
                save_checkpoint(
                    model, optimizer, completed_epoch, monitor,
                    os.path.join(args.checkpoint_dir, f"epoch_{completed_epoch:03d}.pt"),
                    scheduler, scaler, metrics, run_config=run_config, curriculum_state=curriculum_state,
                )
            if monitor <= best_validation_loss:
                best_validation_loss = monitor
                save_checkpoint(
                    model, optimizer, completed_epoch, monitor, os.path.join(args.checkpoint_dir, "best.pt"),
                    scheduler, scaler, metrics, run_config=run_config, curriculum_state=curriculum_state,
                )
                logger.write(f"new best checkpoint loss={monitor:.4f} path={args.checkpoint_dir}/best.pt")

    summary = {
        "status": "complete", "epochs": total_epochs, "best_monitor_loss": best_validation_loss,
        "best_checkpoint": os.path.join(args.checkpoint_dir, "best.pt"), "dataset": args.dataset_path,
        "curriculum": stages,
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.write("run complete " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise
