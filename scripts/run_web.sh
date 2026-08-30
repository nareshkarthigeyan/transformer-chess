#!/usr/bin/env bash
# Run the local presentation interface after training.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python3 app.py
