"""Create a presentation-friendly SQL snapshot of a processed dataset cache."""

from __future__ import annotations

import argparse

from src.data_loader import DEFAULT_DATASET_PATH
from src.sql_export import DEFAULT_SQL_PREVIEW_PATH, DEFAULT_SQL_PREVIEW_ROWS, export_dataset_sql


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", default=DEFAULT_SQL_PREVIEW_PATH)
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_SQL_PREVIEW_ROWS,
        help="Preview rows to export; use 0 for schema-only or -1 for all rows.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.rows < -1:
        raise ValueError("--rows must be -1, 0, or a positive integer")
    rows = None if args.rows == -1 else args.rows
    path = export_dataset_sql(args.dataset, output_path=args.output, preview_rows=rows)
    print(f"SQL preview written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

