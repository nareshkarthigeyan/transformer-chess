import argparse
from pathlib import Path

import chess.pgn

from src import generate_game_gif, generate_game_mp4, generate_game_webm, metadata_from_game


def read_game(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as pgn_file:
        return chess.pgn.read_game(pgn_file)


def output_label(path):
    return path.stem


def generate_from_pgn(path, duration, formats, webm_crf, mp4_crf):
    game = read_game(path)
    if game is None:
        print(f"Skipped unreadable PGN: {path}")
        return None

    moves = list(game.mainline_moves())
    metadata = metadata_from_game(game, source_path=path)
    label = output_label(path)
    outputs = []

    if "gif" in formats:
        outputs.append(
            generate_game_gif(
                moves,
                label,
                duration_per_move=duration,
                metadata=metadata,
            )
        )

    if "webm" in formats:
        outputs.append(
            generate_game_webm(
                moves,
                label,
                duration_per_move=duration,
                metadata=metadata,
                crf=webm_crf,
            )
        )

    if "mp4" in formats:
        outputs.append(
            generate_game_mp4(
                moves,
                label,
                duration_per_move=duration,
                metadata=metadata,
                crf=mp4_crf,
            )
        )

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Generate GIF animations from existing exported PGN files."
    )
    parser.add_argument(
        "--pgn-dir",
        default="exports/pgns",
        help="Directory containing PGN files. Defaults to exports/pgns.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Seconds per move frame. Defaults to 1.0.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of PGNs to process.",
    )
    parser.add_argument(
        "--format",
        choices=["gif", "webm", "mp4", "both", "all"],
        default="both",
        help="Output format to generate. 'both' means GIF and WebM. Defaults to both.",
    )
    parser.add_argument(
        "--webm-crf",
        type=int,
        default=34,
        help="VP9 CRF for WebM output. Higher is smaller. Defaults to 34.",
    )
    parser.add_argument(
        "--mp4-crf",
        type=int,
        default=32,
        help="H.264 CRF for MP4 output. Higher is smaller. Defaults to 32.",
    )
    args = parser.parse_args()

    pgn_dir = Path(args.pgn_dir)
    if not pgn_dir.exists():
        raise SystemExit(f"PGN directory not found: {pgn_dir}")

    pgn_paths = sorted(pgn_dir.glob("*.pgn"))
    if args.limit is not None:
        pgn_paths = pgn_paths[: args.limit]

    if not pgn_paths:
        raise SystemExit(f"No PGN files found in: {pgn_dir}")

    if args.format == "both":
        formats = ["gif", "webm"]
    elif args.format == "all":
        formats = ["gif", "webm", "mp4"]
    else:
        formats = [args.format]

    generated = 0
    for path in pgn_paths:
        outputs = generate_from_pgn(
            path,
            args.duration,
            formats,
            args.webm_crf,
            args.mp4_crf,
        )
        if outputs:
            generated += len(outputs)

    print(f"Generated {generated} file(s) from {pgn_dir}.")


if __name__ == "__main__":
    main()
