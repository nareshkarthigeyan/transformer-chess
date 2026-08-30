# Let's Think Chess — Stockfish-Distilled Geometry Transformer

An end-to-end research/demo pipeline for a compact chess transformer. It learns
from PGN positions using Stockfish policy and value supervision, applies a
learned geometric attention bias, saves resumable checkpoints, and serves a
browser playground with live Stockfish move review and a layer-wise Logic Lens.

The project is deliberately runnable on a free Colab or Kaggle GPU. It is also
honest about measurement: a small corpus cannot guarantee a particular Elo, and
the included benchmark reports a **baseline-relative proxy**, not FIDE/Lichess
rating.

## One-command cloud run

Turn on a **GPU** runtime first. Then paste this single notebook cell—there is
no `%cd`, manual package installation, Stockfish setup, or data download step:

```bash
!if [ -d transformer-chess/.git ]; then git -C transformer-chess pull --ff-only; else git clone https://github.com/nareshkarthigeyan/transformer-chess.git; fi; bash transformer-chess/scripts/cloud_train.sh
```

The script installs Python packages and Stockfish, downloads a diverse bounded
public PGN corpus (five public players × 1,000 games by default) only when
`data/*.pgn` is absent, changes into its own repository directory, builds or
resumes the teacher cache, and trains the `strong` curriculum preset. It needs
no manual Stockfish download or path setup.

To use the project PGNs you already have, place them in `data/` before the last
command. To retain checkpoints across a Colab reset, mount Drive and point the
checkpoint/log outputs at it:

```bash
!bash transformer-chess/scripts/cloud_train.sh \
  --data-dir /content/drive/MyDrive/chess_pgns \
  --checkpoint-dir /content/drive/MyDrive/let_think_chess/checkpoints \
  --log-path /content/drive/MyDrive/let_think_chess/logs/training.log.txt \
  --metrics-path /content/drive/MyDrive/let_think_chess/logs/training_metrics.jsonl
```

The same command works in a Kaggle notebook cell. Kaggle's `/kaggle/working/`
is the appropriate location for outputs. GPU is the tested accelerator path.
TPU requires a matching `torch-xla` installation; it is intentionally not
installed automatically because Colab's XLA builds are runtime/version-specific.

## What `cloud_train.sh` runs

```text
PGN games
  → Stockfish MultiPV teacher labels (policy + value)
  → compressed NumPy cache, resumable as *.partial.npz
  → legality-masked policy/value transformer training
  → last / epoch / best full checkpoints
  → plain-text log + JSONL metrics + JSON summary
```

The default `strong` preset labels up to 200,000 positions with an 80 ms
Stockfish teacher budget and MultiPV=4, then runs a resumable curriculum:
human-move warm-up, Stockfish distillation, opening-aware plus tactical/endgame
oversampling, and hard-example fine-tuning repeated for two rounds. The cache records game IDs so
validation is split by complete games rather than random positions. This is a
bounded same-day job on a free cloud GPU; actual time depends mostly on
Stockfish labelling speed. Use `--preset smoke` to verify a new runtime,
`--preset presentation` for a smaller curriculum, or `--preset research` for a
much slower depth-15 job.

## Outputs and recovery

| Artifact | Purpose |
| --- | --- |
| `data/stockfish_distilled_dataset.npz` | Policy/value cache including legal move masks and board state |
| `data/stockfish_distilled_dataset.partial.npz` | Safe label-generation recovery point |
| `checkpoints/last.pt` | Latest full resumable checkpoint |
| `checkpoints/best.pt` | Lowest validation-loss checkpoint; use this for the app |
| `checkpoints/epoch_*.pt` | Periodic epoch snapshots for ablations or rollback (every 4 epochs by default) |
| `logs/training.log.txt` | Human-readable timestamped training journal |
| `logs/training_metrics.jsonl` | One machine-readable train/validation record per epoch |
| `reports/training_summary.json` | Compact final training summary |

If a Colab session disconnects, rerun exactly the same command. Dataset
labelling continues from the partial cache and training continues from the
last completed curriculum epoch with:

```bash
python run_train.py --preset strong --resume auto
```

## Local training and evaluation

```bash
python -m pip install -r requirements.txt
python run_train.py --preset presentation
python evaluate_elo.py --checkpoint checkpoints/best.pt --media
CHECKPOINT_PATH=checkpoints/best.pt python app.py
```

The app is served at `http://127.0.0.1:5001`. Training in Colab/Kaggle and
running the Flask interface locally is the simplest presentation workflow:
download `checkpoints/best.pt`, then start the last command on your laptop.

`evaluate_elo.py` alternates colours against a uniform-random legal mover and,
when Stockfish is present, a fixed depth-1 Stockfish baseline. It exports PGNs,
optional representative GIF/WebM/MP4 files, `reports/evaluation.json`, and
`logs/evaluation.log.txt`. The numeric result is a reproducible baseline
**proxy**, not official Elo. For the demo, show the record, legal-move rate,
Stockfish centipawn-loss reviews, and the stated evaluation conditions.

## Model and research claims

- **Input:** 64 piece tokens plus side-to-move, castling, en-passant, and clock
  state features.
- **Geometry:** every self-attention layer receives a learned additive bias by
  Manhattan distance between board squares. Its learned values are exposed in
  the Architecture tab.
- **Teacher:** Stockfish supplies a hard best move, top-`k` soft MultiPV move
  distribution, and side-to-move value target.
- **Loss:** legal-masked hard policy cross-entropy mixed with soft teacher
  policy loss, plus weighted value MSE.
- **Legality:** selection always searches `python-chess` legal moves; the model
  cannot play an illegal move.
- **Logic Lens:** applies the trained policy head to input and each encoder
  residual stream, then displays only legal-move scores. It is a post-hoc
  interpretability diagnostic—not model chain-of-thought.

The compact 4,096 policy head represents source/destination squares. Promotion
piece choice is still a known limitation: the UI prefers queen promotion when
promotion choices share a bucket. This does not allow illegal moves, but a
research-grade next revision should use a full promotion-aware action space.

## Presentation-ready web interface

Run `CHECKPOINT_PATH=checkpoints/best.pt python app.py` and open the three tabs:

- **Play:** legality-checked browser chess board, transformer/Stockfish/hybrid
  routing, confidence, value, and entropy.
- **Architecture:** model size, device, checkpoint state, and learned geometry
  bias.
- **Logit lens:** how top legal move scores shift from input projection through
  every encoder layer.

After a human move, the interface requests a fixed-depth live Stockfish review:
best move, centipawn loss, quality label, and an explicitly heuristic quality
band. A single move does not determine a player's Elo, so the UI never claims
that it does.

## Common commands

```bash
# Fast installation/runtime proof
python run_train.py --preset smoke --max-positions 1000

# Rebuild labels after changing PGNs or teacher settings
python run_train.py --preset strong --rebuild-dataset

# Download the default five-player corpus explicitly
python scripts/download_lichess_games.py --games-per-user 1000 --output-dir data

# Inspect one model checkpoint through the UI
CHECKPOINT_PATH=checkpoints/best.pt bash scripts/run_web.sh

# Basic code/model smoke test
python run_sanity_check.py
```

See [CHANGES.md](CHANGES.md) for the implementation and presentation checklist.
