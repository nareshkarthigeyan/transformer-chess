---
license: mit
tags:
  - chess
  - transformer
  - reinforcement-learning
pipeline_tag: reinforcement-learning
---

# Transformer Chess

Transformer Chess is a compact research workspace for training, playing, and
inspecting a transformer-based chess move predictor. The model learns from PGN
positions, masks predictions to legal moves, and can be evaluated against a
random bot or a depth-limited Stockfish baseline.

## What This Project Contains

- A PyTorch transformer model for board-state to move prediction.
- PGN ingestion that converts board positions into cached NumPy arrays.
- Stockfish distillation that labels each position with a teacher policy move
  and side-to-move value target.
- Legal-move masking at inference time through `python-chess`.
- A CLI training script for generating `checkpoint.pt`.
- A CLI evaluator that exports PGN games and representative GIF animations.
- A standalone GIF/WebM/MP4 generator for existing exported PGN files.
- A polished Flask playground for playing the model, inspecting its architecture,
  and probing every encoder layer with a legal-move-masked logit lens.
- An isolated Stockfish provider with Transformer-only, Stockfish-only, and
  transparent 50–70% Hybrid routing modes.
- A sanity-check script for verifying that the training loop can overfit a tiny
  PGN sample.

## Repository Layout

```text
.
|-- app.py                 # Flask playground and tournament API
|-- commands.txt           # Fresh-install commands for macOS, Linux, Windows
|-- evaluate_elo.py        # CLI benchmark runner
|-- generate_gifs.py       # Regenerate GIFs/WebMs/MP4s from exports/pgns
|-- play.py                # Terminal game against the trained model
|-- prepare_dataset.py     # Build cached Stockfish-distilled NumPy dataset
|-- run_sanity_check.py    # Small overfit test for the training loop
|-- run_train.py           # Main training entry point
|-- requirements.txt       # Python dependencies
|-- src/
|   |-- data_loader.py     # PGN parsing, Stockfish distillation, NumPy dataset
|   |-- evaluate.py        # Legal move masked inference
|   |-- exporter.py        # PGN and GIF export
|   |-- model.py           # Transformer architecture
|   |-- tournament.py      # Shared tournament simulation logic
|   `-- train.py           # Training and checkpoint helpers
`-- templates/
    `-- index.html         # Tabbed play / architecture / logit-lens UI
|-- src/providers/
    |-- stockfish.py       # Optional UCI provider
    `-- transformer.py     # Model inference + layer probes
```

## Setup

Use Python 3.10 or newer. A virtual environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project expects PGN files in `data/`. Model checkpoints are written to
`checkpoint.pt` by default. Both are ignored by Git because they are local
artifacts.

## Optional System Dependencies

Stockfish evaluation looks for `STOCKFISH_PATH`, a `stockfish` executable on
`PATH`, and common macOS/Linux locations:

```bash
brew install stockfish
```

On Windows, install the binary and set `STOCKFISH_PATH` to its full path before
starting the app.

If Stockfish is not installed, random-bot evaluation still works.

## Training

Place PGN files in `data/`, then build the cached Stockfish-distilled NumPy
dataset:

```bash
python prepare_dataset.py
```

By default, this writes `data/stockfish_distilled_dataset.npz`, labels all PGN
positions with Stockfish depth 15, stores the Stockfish best move as the policy
target, and stores a side-to-move value target in `[-1, 1]`.

Useful distillation options:

```bash
python prepare_dataset.py --stockfish-depth 15
python prepare_dataset.py --stockfish-time 0.1
python prepare_dataset.py --max-games-per-file 200
```

`--stockfish-time` is measured in seconds and overrides depth. Use `0.1` for a
100 ms per-position teacher budget.

After the cache exists, train from the NumPy arrays:

```bash
python run_train.py
```

`run_train.py` automatically builds the cache if it is missing. Pass
`--rebuild-dataset` to force regeneration after changing PGNs or teacher
settings:

```bash
python run_train.py --rebuild-dataset --stockfish-depth 15
python run_train.py --dataset-path data/stockfish_distilled_dataset.npz --epochs 10
```

The model trains with an AlphaZero-style policy head plus value head. Policy
loss distills Stockfish's best legal move, while value loss distills the
Stockfish evaluation for the side to move.

## Evaluation

Run the benchmark suite after training:

```bash
python evaluate_elo.py
```

The evaluator:

- Loads `checkpoint.pt`.
- Runs 20 games against the random bot.
- Runs 20 games against Stockfish depth 1 when Stockfish is available.
- Saves PGN files to `exports/pgns/`.
- Saves representative GIFs to `exports/gifs/`.
- Saves compressed WebM videos to `exports/webm/`.
- Saves compressed MP4 videos to `exports/mp4/`.
- Prints a consolidated Elo-style performance estimate.

GIFs use a blue board, filled chess-piece glyphs, and a footer that shows match
number, opponent type, result, side color swatches, ply count, and last move.

## Regenerate GIFs, WebMs, And MP4s From PGNs

To generate GIF and WebM files from PGNs that already exist in `exports/pgns/`,
run the default command:

```bash
python generate_gifs.py
```

Useful options:

```bash
python generate_gifs.py --pgn-dir exports/pgns --duration 1
python generate_gifs.py --limit 3
python generate_gifs.py --format webm --webm-crf 34
python generate_gifs.py --format mp4 --mp4-crf 32
python generate_gifs.py --format all
```

WebM output uses VP9 with constant-quality compression. Lower CRF values produce
larger, cleaner files; higher CRF values produce smaller files. `34` is a good
blog-friendly default for this board animation style.

MP4 output uses H.264 with constant-quality compression and faststart metadata.
`32` is the default MP4 CRF and is intended to keep typical generated games in
the 500 KB to 1 MB range or lower while preserving readable pieces and text.

## Web Dashboard

Start the Flask dashboard:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5001
```

The dashboard runs a background tournament thread and polls the current board
state, move log, and final benchmark metrics.

The main interface has three tabs:

- Playground: click-to-move chess board, move history, live value, confidence,
  entropy, and provider routing.
- Architecture: the token, encoder, policy-head, and value-head shape loaded
  from the model.
- Logit lens: the policy head projected over the input stream and each encoder
  layer, restricted to legal moves.

Exact fresh-install commands for macOS, Linux, and Windows are in
`commands.txt`. The checkpoint is downloaded from the Hugging Face model repo
instead of being committed to Git.

Stockfish assistance is implemented as a standalone provider so it can be
removed later. Hybrid mode keeps the assist rate in the 50–70% range and labels
the provider used for each engine move in the UI.

## Play In The Terminal

After training, you can play White against the model:

```bash
python play.py
```

Enter moves in UCI format, for example `e2e4` or `g1f3`. Type `quit` to exit.

## Sanity Check

Before running a full training job, validate the basic training loop:

```bash
python run_sanity_check.py
```

This script builds a tiny dataset from one PGN string and verifies that the
model, loss function, optimizer, and data conversion path work together.

## Notes On The Model

The current model encodes each board as 64 tokens:

- `0` for empty squares.
- `1` through `6` for white pawn, knight, bishop, rook, queen, and king.
- `7` through `12` for black pawn, knight, bishop, rook, queen, and king.

The output space is `4096`, representing `from_square * 64 + to_square`.
Promotion piece choice is not modeled separately yet; this is a known
simplification.

## Generated Files

The following are treated as local artifacts and should not be committed:

- `checkpoint.pt`
- `data/`
- `exports/`
- `__pycache__/`
- virtual environments and tool caches

## Common Issues

`checkpoint.pt` missing:
Run `python run_train.py` first.

Stockfish binary not found:
Install Stockfish or use the random-bot benchmark.

NumPy and Matplotlib binary mismatch:
The exporter does not depend on Matplotlib or Cairo. Reinstall dependencies in a
clean virtual environment if your global Python environment has incompatible
compiled packages.
