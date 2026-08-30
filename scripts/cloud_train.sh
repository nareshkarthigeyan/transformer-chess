#!/usr/bin/env bash
# One-command Colab/Kaggle GPU setup + Stockfish distillation + checkpointed training.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq stockfish
fi

python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -r requirements.txt

if [[ -z "${STOCKFISH_PATH:-}" ]]; then
  for candidate in /usr/games/stockfish /usr/bin/stockfish "$(command -v stockfish 2>/dev/null || true)"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      export STOCKFISH_PATH="$candidate"
      break
    fi
  done
fi

if [[ -z "${STOCKFISH_PATH:-}" ]]; then
  echo "Stockfish was not found after installation. Set STOCKFISH_PATH and retry." >&2
  exit 1
fi

shopt -s nullglob
pgn_files=(data/*.pgn)
if (( ${#pgn_files[@]} == 0 )); then
  echo "No local PGNs found; downloading a bounded public Lichess corpus."
  python3 scripts/download_lichess_games.py \
    --user "${LICHESS_USER:-DrNykterstein}" \
    --max-games "${LICHESS_MAX_GAMES:-800}" \
    --output data/lichess_public_games.pgn
fi

python3 run_train.py --preset presentation "$@"
