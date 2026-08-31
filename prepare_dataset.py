import argparse
import os

from src.data_loader import DEFAULT_DATASET_PATH, build_stockfish_distilled_dataset
from src.sql_export import DEFAULT_SQL_PREVIEW_PATH, DEFAULT_SQL_PREVIEW_ROWS, export_dataset_sql


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a cached NumPy dataset with Stockfish policy/value labels."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--max-games-per-file", type=int, default=None)
    parser.add_argument("--stockfish-path", default=os.environ.get("STOCKFISH_PATH"))
    parser.add_argument("--stockfish-depth", type=int, default=15)
    parser.add_argument(
        "--stockfish-time",
        type=float,
        default=None,
        help="Seconds per Stockfish evaluation. Overrides --stockfish-depth.",
    )
    parser.add_argument("--progress-interval", type=int, default=500)
    parser.add_argument("--sql-preview-path", default=DEFAULT_SQL_PREVIEW_PATH)
    parser.add_argument("--sql-preview-rows", type=int, default=DEFAULT_SQL_PREVIEW_ROWS)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = build_stockfish_distilled_dataset(
        data_dir=args.data_dir,
        output_path=args.output,
        max_games_per_file=args.max_games_per_file,
        stockfish_path=args.stockfish_path,
        stockfish_depth=args.stockfish_depth,
        stockfish_time=args.stockfish_time,
        progress_interval=args.progress_interval,
    )
    if args.sql_preview_rows < 0:
        raise ValueError("--sql-preview-rows must be non-negative")
    export_dataset_sql(
        dataset_path,
        output_path=args.sql_preview_path,
        preview_rows=args.sql_preview_rows,
    )


if __name__ == "__main__":
    main()
