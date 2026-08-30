# End-to-end presentation upgrade

## Added

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
