import argparse
import os

from src.data_loader import DEFAULT_DATASET_PATH, build_stockfish_distilled_dataset


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
    return parser.parse_args()


def main():
    args = parse_args()
    build_stockfish_distilled_dataset(
        data_dir=args.data_dir,
        output_path=args.output,
        max_games_per_file=args.max_games_per_file,
        stockfish_path=args.stockfish_path,
        stockfish_depth=args.stockfish_depth,
        stockfish_time=args.stockfish_time,
        progress_interval=args.progress_interval,
    )


if __name__ == "__main__":
    main()
