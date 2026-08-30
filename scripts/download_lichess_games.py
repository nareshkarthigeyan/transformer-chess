"""Download a diverse bounded Lichess PGN corpus without extra packages.

The Lichess user export endpoint streams many PGNs in one response. We keep one
file per player so the dataset builder can interleave players instead of
consuming one account's games first. Failed accounts are reported and skipped;
the run fails only when every requested account fails.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_USERS = (
    "DrNykterstein",
    "Hikaru",
    "DanielNaroditsky",
    "Bortnyk",
    "ChessNetwork",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Download diverse public games from the Lichess API.")
    parser.add_argument(
        "--users", nargs="+", default=list(DEFAULT_USERS),
        help="Public Lichess usernames (default: five strong, stylistically varied accounts).",
    )
    parser.add_argument(
        "--user", dest="users_compat", default=None,
        help="Backward-compatible alias for downloading one user.",
    )
    parser.add_argument(
        "--games-per-user", "--max-games", dest="games_per_user", type=int, default=1_000,
    )
    parser.add_argument("--output-dir", default="data", help="Directory for one PGN file per user.")
    parser.add_argument("--output", default=None, help="Legacy single-file output; combines all successful users.")
    parser.add_argument("--perf-type", default="blitz,rapid,classical", help="Lichess performance filters.")
    return parser.parse_args(argv)


def _safe_filename(user: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", user).strip("._") or "lichess_user"


def _download_user(user: str, max_games: int, perf_type: str) -> bytes:
    url = (
        f"https://lichess.org/api/games/user/{user}?max={max_games}"
        f"&pgnInJson=false&perfType={perf_type}&clocks=false&evals=false&opening=true"
    )
    request = Request(
        url,
        headers={
            "Accept": "application/x-chess-pgn",
            "User-Agent": "transformer-chess-research/2.0",
        },
    )
    try:
        with urlopen(request, timeout=180) as response:
            content = response.read()
    except HTTPError as exc:
        raise RuntimeError(f"Lichess returned HTTP {exc.code} for user {user}.") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Lichess for {user}: {exc.reason}") from exc
    if len(content) < 100 or b"[Event" not in content:
        raise RuntimeError(f"Lichess returned no readable PGN games for {user}.")
    return content


def main(argv=None):
    args = parse_args(argv)
    users = [args.users_compat] if args.users_compat else args.users
    users = list(dict.fromkeys(users))
    if args.games_per_user < 1:
        raise ValueError("--games-per-user must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    successful = []
    failures = []
    for user in users:
        try:
            content = _download_user(user, args.games_per_user, args.perf_type)
        except RuntimeError as exc:
            failures.append(str(exc))
            print(f"Warning: {exc}", file=sys.stderr)
            continue
        filename = output_dir / f"lichess_{_safe_filename(user)}.pgn"
        filename.write_bytes(content)
        game_count = content.count(b"[Event ")
        successful.append((filename, content))
        print(f"Downloaded {game_count:,} games ({len(content):,} bytes) for {user} → {filename}")

    if not successful:
        raise RuntimeError("All requested Lichess accounts failed. " + " | ".join(failures))
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\n\n".join(content for _, content in successful))
        print(f"Combined corpus written to {destination}")
    print(f"Corpus ready: {sum(content.count(b'[Event ') for _, content in successful):,} games from {len(successful)} accounts.")
    if failures:
        print(f"Skipped {len(failures)} account(s); continue with the successful sources.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        raise
