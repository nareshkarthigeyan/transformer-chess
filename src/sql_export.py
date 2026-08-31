"""Export the model-ready NumPy cache as a readable SQLite SQL snapshot.

The training pipeline keeps the full dataset in a compact ``.npz`` file.  This
module produces a bounded SQL dump for inspection and presentations, showing
what a processed FEN looks like immediately before it enters the DataLoader.
The dump is intentionally a preview by default, not a second copy of the
entire training corpus.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


DEFAULT_SQL_PREVIEW_PATH = "processed_dataset.sql"
DEFAULT_SQL_PREVIEW_ROWS = 100


def _sql_value(value: Any) -> str:
    """Render a Python value as a safe, readable SQLite literal."""
    if value is None:
        return "NULL"
    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return "NULL" if not np.isfinite(number) else repr(number)
    text = str(value).replace("'", "''")
    return f"'{text}'"


def _json(value: Any) -> str:
    """Serialize NumPy arrays/scalars into compact JSON text."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _array(data: Any, key: str, length: int, default: Any) -> Iterable[Any]:
    if key in data.files:
        return data[key]
    if callable(default):
        return [default(index) for index in range(length)]
    return [default] * length


def _metadata_rows(metadata: dict, total_positions: int, preview_rows: int):
    rows = {
        "dataset_version": metadata.get("dataset_version"),
        "status": metadata.get("status", "unknown"),
        "total_positions": total_positions,
        "preview_rows": preview_rows,
        **metadata,
    }
    for key in sorted(rows):
        yield key, rows[key]


def export_dataset_sql(
    dataset_path: str,
    output_path: str = DEFAULT_SQL_PREVIEW_PATH,
    preview_rows: Optional[int] = DEFAULT_SQL_PREVIEW_ROWS,
) -> str:
    """Write a SQLite-compatible SQL preview and return its path.

    ``preview_rows=None`` exports every row and should only be used for a
    deliberately small dataset.  The normal bounded preview keeps the root
    folder useful for git, Colab output, and a project presentation.
    """
    if preview_rows is not None and preview_rows < 0:
        raise ValueError("preview_rows must be non-negative or None")
    with np.load(dataset_path, allow_pickle=False) as data:
        if "fens" not in data.files or "boards" not in data.files:
            raise ValueError("Dataset cache must contain processed fens and board encodings")
        total_positions = int(len(data["fens"]))
        limit = total_positions if preview_rows is None else min(int(preview_rows), total_positions)
        metadata = {}
        if "metadata" in data.files:
            raw_metadata = data["metadata"].item()
            metadata = json.loads(str(raw_metadata)) if raw_metadata else {}

        state_features = _array(data, "state_features", total_positions, lambda _index: [0, 0, 0, 0, 0])
        human_ids = _array(data, "human_move_ids", total_positions, None)
        teacher_ids = _array(data, "teacher_move_ids", total_positions, None)
        values = _array(data, "values", total_positions, None)
        legal_ids = _array(data, "legal_move_ids", total_positions, lambda _index: [])
        top_ids = _array(data, "teacher_top_move_ids", total_positions, lambda _index: [])
        top_probs = _array(data, "teacher_top_probs", total_positions, lambda _index: [])
        source_files = _array(data, "source_files", total_positions, "unknown")
        game_ids = _array(data, "game_ids", total_positions, lambda index: str(index))
        eco_codes = _array(data, "eco_codes", total_positions, "UNK")
        priority_flags = _array(data, "priority_flags", total_positions, lambda _index: [0, 0, 0])
        sampling_weights = _array(data, "sampling_weights", total_positions, 1.0)

        lines = [
            "-- Transformer Chess: model-ready dataset preview",
            f"-- Source cache: {dataset_path}",
            f"-- Preview rows: {limit} of {total_positions}",
            "-- Import with: sqlite3 processed_dataset.db < processed_dataset.sql",
            "PRAGMA foreign_keys = ON;",
            "BEGIN TRANSACTION;",
            "",
            "DROP TABLE IF EXISTS teacher_policy;",
            "DROP TABLE IF EXISTS legal_move_mask;",
            "DROP TABLE IF EXISTS processed_positions;",
            "DROP TABLE IF EXISTS dataset_metadata;",
            "",
            "CREATE TABLE dataset_metadata (key TEXT PRIMARY KEY, value TEXT);",
            "CREATE TABLE processed_positions (",
            "    position_id INTEGER PRIMARY KEY,",
            "    fen TEXT NOT NULL,",
            "    board_tokens_json TEXT NOT NULL,",
            "    state_features_json TEXT NOT NULL,",
            "    human_move_id INTEGER,",
            "    teacher_move_id INTEGER,",
            "    teacher_value REAL,",
            "    source_file TEXT,",
            "    game_id TEXT,",
            "    eco_code TEXT,",
            "    tactical INTEGER NOT NULL DEFAULT 0,",
            "    endgame INTEGER NOT NULL DEFAULT 0,",
            "    opening INTEGER NOT NULL DEFAULT 0,",
            "    sampling_weight REAL NOT NULL DEFAULT 1.0",
            ");",
            "CREATE TABLE legal_move_mask (",
            "    position_id INTEGER NOT NULL REFERENCES processed_positions(position_id),",
            "    move_rank INTEGER NOT NULL,",
            "    move_id INTEGER NOT NULL,",
            "    PRIMARY KEY (position_id, move_rank)",
            ");",
            "CREATE TABLE teacher_policy (",
            "    position_id INTEGER NOT NULL REFERENCES processed_positions(position_id),",
            "    policy_rank INTEGER NOT NULL,",
            "    move_id INTEGER NOT NULL,",
            "    probability REAL NOT NULL,",
            "    PRIMARY KEY (position_id, policy_rank)",
            ");",
            "",
        ]

        for key, value in _metadata_rows(metadata, total_positions, limit):
            lines.append(
                "INSERT INTO dataset_metadata(key, value) VALUES "
                f"({_sql_value(key)}, {_sql_value(_json(value) if not isinstance(value, str) else value)});"
            )

        for position_id in range(limit):
            flags = list(priority_flags[position_id]) if priority_flags is not None else [0, 0, 0]
            flags = (flags + [0, 0, 0])[:3]
            lines.append(
                "INSERT INTO processed_positions("
                "position_id, fen, board_tokens_json, state_features_json, human_move_id, "
                "teacher_move_id, teacher_value, source_file, game_id, eco_code, tactical, "
                "endgame, opening, sampling_weight) VALUES ("
                + ", ".join(
                    _sql_value(value)
                    for value in (
                        position_id,
                        data["fens"][position_id],
                        _json(data["boards"][position_id]),
                        _json(state_features[position_id]),
                        human_ids[position_id],
                        teacher_ids[position_id],
                        values[position_id],
                        source_files[position_id],
                        game_ids[position_id],
                        eco_codes[position_id],
                        flags[0],
                        flags[1],
                        flags[2],
                        sampling_weights[position_id],
                    )
                )
                + ");"
            )

            for rank, move_id in enumerate(np.asarray(legal_ids[position_id]).reshape(-1).tolist()):
                if int(move_id) < 0:
                    continue
                lines.append(
                    "INSERT INTO legal_move_mask(position_id, move_rank, move_id) VALUES "
                    f"({position_id}, {rank}, {int(move_id)});"
                )

            position_top_ids = np.asarray(top_ids[position_id]).reshape(-1).tolist()
            position_top_probs = np.asarray(top_probs[position_id]).reshape(-1).tolist()
            for rank, move_id in enumerate(position_top_ids):
                if int(move_id) < 0:
                    continue
                probability = position_top_probs[rank] if rank < len(position_top_probs) else 0.0
                lines.append(
                    "INSERT INTO teacher_policy(position_id, policy_rank, move_id, probability) VALUES "
                    f"({position_id}, {rank}, {int(move_id)}, {_sql_value(probability)});"
                )

        lines.extend(["", "COMMIT;", ""])
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    return os.fspath(output_path)

