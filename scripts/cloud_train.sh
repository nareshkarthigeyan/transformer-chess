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

data_dir="${DATA_DIR:-data}"
has_data_arg=0
for ((arg_index = 1; arg_index <= $#; arg_index++)); do
  arg_value="${!arg_index}"
  case "$arg_value" in
    --data-dir)
      next_index=$((arg_index + 1))
      if (( next_index <= $# )); then
        data_dir="${!next_index}"
        has_data_arg=1
      fi
      ;;
    --data-dir=*)
      data_dir="${arg_value#*=}"
      has_data_arg=1
      ;;
  esac
done

shopt -s nullglob
pgn_files=("$data_dir"/*.pgn)
if (( ${#pgn_files[@]} == 0 )); then
  echo "No local PGNs found; downloading a diverse public Lichess corpus."
  python3 scripts/download_lichess_games.py \
    --users ${LICHESS_USERS:-DrNykterstein Hikaru DanielNaroditsky Bortnyk ChessNetwork} \
    --games-per-user "${LICHESS_GAMES_PER_USER:-1000}" \
    --output-dir "$data_dir"
elif [[ -f "$data_dir/lichess_public_games.pgn" && ! -f "$data_dir/.diverse_corpus_v2" && "${AUTO_DIVERSE_CORPUS:-1}" == "1" ]]; then
  echo "Legacy single-user PGN detected; adding the diverse multi-user corpus."
  python3 scripts/download_lichess_games.py \
    --users ${LICHESS_USERS:-DrNykterstein Hikaru DanielNaroditsky Bortnyk ChessNetwork} \
    --games-per-user "${LICHESS_GAMES_PER_USER:-1000}" \
    --output-dir "$data_dir"
fi

touch "$data_dir/.diverse_corpus_v2"
train_args=("$@")
if (( has_data_arg == 0 )) && [[ -n "${DATA_DIR:-}" ]]; then
  train_args+=(--data-dir "$data_dir")
fi
python3 run_train.py --preset "${TRAIN_PRESET:-strong}" "${train_args[@]}"
