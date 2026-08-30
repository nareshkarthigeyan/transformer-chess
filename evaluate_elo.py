"""Repeatable baseline matches and an honestly-labelled rating proxy."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import chess
import torch

from src import (
    ChessTransformer,
    RANDOM_BASE_ELO,
    STOCKFISH_DEPTH_1_BASE_ELO,
    generate_game_gif,
    generate_game_mp4,
    generate_game_webm,
    load_checkpoint_weights,
    open_stockfish,
    play_single_game,
    save_game_pgn,
    summarize_results,
)


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_from_checkpoint(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ChessTransformer.from_config(checkpoint.get("model_config", {})).to(device)
    load_checkpoint_weights(model, path, device)
    model.eval()
    return model


def run_tournament(model, device, opponent_type="random", games=20, max_moves=150, make_media=False):
    wins, losses, draws = 0, 0, 0
    saved_matches = {"win": None, "draw": None, "lose": None}
    engine = open_stockfish(opponent_type)
    label = "Stockfish depth 1" if opponent_type == "stockfish_depth_1" else "uniform random legal mover"
    print(f"\nBenchmark: model vs {label} | games={games} | max plies={max_moves}")

    try:
        for game_num in range(games):
            model_color = chess.WHITE if game_num % 2 == 0 else chess.BLACK
            game = play_single_game(
                model=model, device=device, opponent_type=opponent_type,
                model_color=model_color, engine=engine, max_moves=max_moves,
            )
            if game.category == "win":
                wins += 1
            elif game.category == "lose":
                losses += 1
            else:
                draws += 1
            white_player = "Model" if model_color == chess.WHITE else opponent_type
            black_player = "Model" if model_color == chess.BLACK else opponent_type
            metadata = {
                "event": f"Transformer Chess benchmark #{game_num + 1}",
                "white": white_player, "black": black_player, "result": game.result,
                "opponent": opponent_type,
            }
            save_game_pgn(
                game.board.move_stack, white_player, black_player, game.result,
                game_num + 1, opponent_type,
            )
            if saved_matches[game.category] is None:
                saved_matches[game.category] = {"moves": list(game.board.move_stack), "metadata": metadata}
            print(
                f"game={game_num + 1:02d} side={'W' if model_color else 'B'} "
                f"result={game.result} plies={game.move_count} ending={game.reason}"
            )
    finally:
        if engine:
            engine.quit()

    if make_media:
        for category, match in saved_matches.items():
            if match is None:
                continue
            output_label = f"model_{category}_{opponent_type}"
            generate_game_gif(match["moves"], output_label, metadata=match["metadata"])
            generate_game_webm(match["moves"], output_label, metadata=match["metadata"])
            generate_game_mp4(match["moves"], output_label, metadata=match["metadata"])
    return summarize_results(wins, losses, draws, opponent_type, games)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a saved transformer checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-moves", type=int, default=150)
    parser.add_argument("--skip-stockfish", action="store_true")
    parser.add_argument("--media", action="store_true", help="Also generate one GIF/WebM/MP4 per result class.")
    parser.add_argument("--report", default="reports/evaluation.json")
    parser.add_argument("--log-path", default="logs/evaluation.log.txt")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    device = select_device()
    model = model_from_checkpoint(args.checkpoint, device)
    Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
    random_results = run_tournament(model, device, "random", args.games, args.max_moves, args.media)
    stockfish_results = None
    if not args.skip_stockfish:
        try:
            stockfish_results = run_tournament(
                model, device, "stockfish_depth_1", args.games, args.max_moves, args.media
            )
        except FileNotFoundError as exc:
            print(f"Stockfish benchmark skipped: {exc}")

    proxy_values = [random_results["Elo"]]
    if stockfish_results:
        proxy_values.append(stockfish_results["Elo"])
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "device": str(device),
        "games_per_baseline": args.games,
        "random_legal_baseline": random_results,
        "stockfish_depth_1_baseline": stockfish_results,
        "baseline_relative_proxy": round(sum(proxy_values) / len(proxy_values)),
        "interpretation": (
            "Baseline-relative proxy only, derived from fixed toy baseline assumptions. "
            "It is not an official FIDE/Lichess Elo and should not be presented as one."
        ),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    line = json.dumps(report, sort_keys=True)
    print("\nEvaluation report: " + line)
    with open(args.log_path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return report


if __name__ == "__main__":
    main()
