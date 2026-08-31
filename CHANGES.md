# End-to-end presentation upgrade

## Interactive presentation polish

- White and black pieces now use explicit contrasting fills and shadows instead
  of inheriting the same glyph color.
- Provider and opening-side controls are presented as pre-game setup; the
  active provider is shown in the readout while playing, with settings
  available on demand for the next game.
- Added a mid-game switch-side control that preserves the board and lets the
  engine make the intervening move when needed.
- Added a compact in-game Logic Lens alongside the full Lens tab.
- Added an automatic root-level `processed_dataset.sql` snapshot. It is a
  bounded, SQLite-compatible preview of processed FENs, board/state features,
  legal-move masks, and Stockfish teacher policy rows generated immediately
  before training.
- Added a dark, color-coded browser layout with clearer board coordinates,
  selected/last-move/check highlights, and responsive panels.
- Interactive games now continue through claimable threefold/50-move draws;
  forced checkmate, stalemate, insufficient-material, 75-move, and fivefold
  endings are explained directly in the UI. Users can claim a draw explicitly.
- Added Stockfish centipawn-loss move review with an explicitly non-official
  quality-band diagnostic.
- Added a live chess.com-style Stockfish position evaluation bar. It reports
  the current white-perspective score (including mate), shows a side-to-side
  advantage percentage, and plots centipawn evaluation after every position.
- Undo now keeps the move-quality chart synchronized with the board history.

## Added

- Diverse five-player Lichess bootstrap (up to 5,000 games), one PGN per
  source, with partial-failure tolerance and a backward-compatible `--user`
  option.
- Strong default curriculum: 80 ms Stockfish labels, complete-game
  train/validation split, tactical/endgame-aware weighted sampling, human move
  warm-up, two repeated distillation rounds, and model-driven hard-example
  reweighting.
- Dataset schema v3 stores game IDs, ECO codes, priority flags, and sampling
  weights. Hard-example weights are checkpoint-directory `.npy` artifacts and
  all stage/epoch state is embedded in `last.pt` and `best.pt`.
- Parallel labelling uses independent persistent Stockfish subprocesses from
  CPU worker threads (`--label-workers`); board encodings are cached once so
  the GPU is not wasted repeating Python/chess preprocessing during training.

- `scripts/cloud_train.sh`: one-command Colab/Kaggle GPU installer, Stockfish
  provisioner, optional public PGN bootstrapper, dataset builder, and trainer.
- `scripts/download_lichess_games.py`: bounded public-PGN download with no new
  Python dependency.
- Resumable Stockfish dataset creation. `*.partial.npz` is atomically refreshed
  every 2,000 labelled positions, and the next invocation skips completed rows.
- Full `last.pt`, `best.pt`, and periodic epoch checkpoints including optimizer,
  scheduler, AMP scaler, model configuration, metrics, and run configuration.
- `logs/training.log.txt`, JSONL epoch metrics, a training summary, evaluation
  report, and evaluation log for research-output evidence.
- Mixed hard best-move + soft MultiPV Stockfish policy distillation alongside a
  side-to-move value target.
- State features for side-to-move, castling, en-passant, and move clocks.
- A learned Manhattan-distance attention bias in every encoder layer.
- Live Stockfish centipawn-loss review in the web app.
- Logic Lens disclaimer and geometry-bias display in the web app.

## Corrected

- CUDA is now selected automatically in Colab/Kaggle; the old training script
  only attempted Apple MPS or CPU.
- All supplied training labels are legality-masked before policy loss.
- Existing old checkpoints load compatible tensors safely instead of crashing on
  changed architecture shapes.
- Evaluation calls its rating a baseline-relative proxy. It is not represented
  as official Elo.

## Before presenting

1. Run `bash scripts/cloud_train.sh` on a GPU. Keep Drive/Kaggle output copies.
2. Run `python evaluate_elo.py --checkpoint checkpoints/best.pt --media`.
3. Copy `logs/training.log.txt`, `reports/training_summary.json`, and
   `reports/evaluation.json` into the submission folder.
4. Run `CHECKPOINT_PATH=checkpoints/best.pt python app.py` locally and open the
   Play, Architecture, and Logit Lens tabs before the presentation.
5. State the data source, Stockfish budget, position count, checkpoint epoch,
   and evaluation baseline. Do not claim a verified 600–700 Elo unless an
   external rated match protocol actually supports it.

## Next research-grade improvement

Replace the compact source/destination action head with a promotion-aware move
vocabulary (for example AlphaZero's 4,672 actions). The current system keeps
play legal by choosing from `python-chess` legal moves, but queen promotion is
preferred when promotion variants share the same source/destination bucket.
