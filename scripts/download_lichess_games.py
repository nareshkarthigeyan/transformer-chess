"""Download a bounded public Lichess PGN corpus without extra packages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Download public games from the Lichess API.")
    parser.add_argument("--user", default="DrNykterstein", help="Public Lichess username.")
    parser.add_argument("--max-games", type=int, default=800)
    parser.add_argument("--output", default="data/lichess_public_games.pgn")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.max_games < 1:
        raise ValueError("--max-games must be positive")
    url = f"https://lichess.org/api/games/user/{args.user}?max={args.max_games}&pgnInJson=false"
    request = Request(url, headers={"Accept": "application/x-chess-pgn", "User-Agent": "transformer-chess-research/1.0"})
    try:
        with urlopen(request, timeout=90) as response:
            content = response.read()
    except HTTPError as exc:
        raise RuntimeError(f"Lichess returned HTTP {exc.code} for user {args.user}.") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach the Lichess public API: {exc.reason}") from exc
    if len(content) < 100:
        raise RuntimeError("Lichess returned an empty or invalid PGN response.")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    print(f"Downloaded {len(content):,} bytes of public PGN data to {destination}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        raise
