import torch
import chess

from src import (
    ChessTransformer,
    NUM_GAMES,
    RANDOM_BASE_ELO,
    STOCKFISH_DEPTH_1_BASE_ELO,
    generate_game_gif,
    generate_game_mp4,
    generate_game_webm,
    open_stockfish,
    play_single_game,
    save_game_pgn,
    summarize_results,
)


def run_tournament(model, device, opponent_type="random"):
    wins, losses, draws = 0, 0, 0
    saved_matches = {"win": None, "draw": None, "lose": None}

    try:
        engine = open_stockfish(opponent_type)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return None

    if opponent_type == "stockfish_depth_1":
        print(
            "\nRunning match: ChessTransformer vs. "
            f"Stockfish D1 [Base ELO: {STOCKFISH_DEPTH_1_BASE_ELO}]"
        )
    else:
        print(
            "\nRunning match: ChessTransformer vs. "
            f"Pure Random Bot [Base ELO: {RANDOM_BASE_ELO}]"
        )

    try:
        print("-" * 75)
        print(
            f"{'Game':<6} | {'Model Side':<10} | {'Moves':<6} | "
            f"{'Result':<8} | {'Ending Condition'}"
        )
        print("-" * 75)

        for game_num in range(NUM_GAMES):
            model_color = chess.WHITE if game_num % 2 == 0 else chess.BLACK
            game = play_single_game(
                model=model,
                device=device,
                opponent_type=opponent_type,
                model_color=model_color,
                engine=engine,
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
                "event": f"Transformer Benchmark Match - Game #{game_num + 1}",
                "game_num": str(game_num + 1),
                "white": white_player,
                "black": black_player,
                "result": game.result,
                "opponent": opponent_type,
            }
            save_game_pgn(
                game.board.move_stack,
                white_player,
                black_player,
                game.result,
                game_num + 1,
                opponent_type,
            )

            if saved_matches[game.category] is None:
                saved_matches[game.category] = {
                    "moves": list(game.board.move_stack),
                    "metadata": metadata,
                }

            color_str = "White" if model_color == chess.WHITE else "Black"
            print(
                f"#{game_num + 1:<4} | {color_str:<10} | "
                f"{game.move_count:<6} | {game.result:<8} | {game.reason}"
            )
    finally:
        if engine:
            engine.quit()

    print(f"\nCompiling match target showcase animations for {opponent_type}...")
    for category, match in saved_matches.items():
        if match is not None:
            label = f"model_{category}_{opponent_type}"
            generate_game_gif(
                match["moves"],
                label,
                metadata=match["metadata"],
            )
            generate_game_webm(
                match["moves"],
                label,
                metadata=match["metadata"],
            )
            generate_game_mp4(
                match["moves"],
                label,
                metadata=match["metadata"],
            )
        else:
            print(
                "No match matched the profile: "
                f"{category.upper()} against {opponent_type}"
            )

    return summarize_results(wins, losses, draws, opponent_type, NUM_GAMES)


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ChessTransformer().to(device)

    try:
        checkpoint = torch.load("checkpoint.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        random_results = run_tournament(model, device, opponent_type="random")
        sf_results = run_tournament(model, device, opponent_type="stockfish_depth_1")

        if random_results is None or sf_results is None:
            raise SystemExit(1)

        random_elo = random_results["Elo"]
        sf_elo = sf_results["Elo"]
        final_estimated_elo = int((random_elo + sf_elo) / 2)

        print("\n" + "=" * 70)
        print("                 FINAL PROJECT REPORT METRICS")
        print("=" * 70)
        print("Opponent Agent            | Record (W-L-D) | Win %  | Calculated Performance")
        print("-" * 70)
        print(
            "Pure Random Bot (Elo ~100) |   "
            f"{random_results['W']}-{random_results['L']}-{random_results['D']:<5}  | "
            f"{random_results['Win%']:.1f}%  | {random_elo} Elo"
        )
        print(
            "Stockfish D1 (Elo ~600)   |   "
            f"{sf_results['W']}-{sf_results['L']}-{sf_results['D']:<5}  | "
            f"{sf_results['Win%']:.1f}%   | {sf_elo} Elo"
        )
        print("-" * 70)
        print(f"Consolidated model performance estimate: {final_estimated_elo} Elo")
        print("=" * 70)

    except FileNotFoundError:
        print("Error: 'checkpoint.pt' missing. Train the model first.")
