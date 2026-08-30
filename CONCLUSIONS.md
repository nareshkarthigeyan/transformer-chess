# Training and evaluation conclusions

Generated from the checkpoint currently available at
`checkpoints/best.pt` and the matching training log on 31 August 2026.

## What was actually evaluated

The checkpoint embeds the following run configuration:

- 30,000 labelled positions from one PGN source (`lichess_public_games.pgn`)
- Stockfish MultiPV=4 with a 30 ms per-position budget
- Four transformer layers, width 128, 35,460,752 parameters
- 24 training epochs with an 8% position-level validation split

The repository also contains `checkpoints/training_summary-2.json`, which
mentions a separate 30-epoch run and `checkpoints_presentation/best.pt`. That
checkpoint is not present, so the results below must not be attributed to that
unavailable run.

## Training conclusions

Validation improved quickly and then stopped improving:

| checkpoint | validation loss | validation top-1 |
| --- | ---: | ---: |
| epoch 1 | 3.4371 | 16.4% |
| epoch 2 | 3.3980 | 19.3% |
| **epoch 3 (best.pt)** | **3.1144** | **21.9%** |
| epoch 24 | 3.4628 | 26.1% |

Training top-1 accuracy continued to rise to 93.6% by epoch 24 while the
validation loss worsened. This is evidence of overfitting to the small,
single-source dataset. `best.pt` is correctly the safer model to demonstrate;
the final epoch is not automatically the best model.

## Match evaluation

The included 20-game benchmark was run with a 150-ply cap per game:

| opponent | W | L | D | score | project proxy |
| --- | ---: | ---: | ---: | ---: | ---: |
| uniform random legal mover | 3 | 0 | 17 | 57.5% | 152 |
| Stockfish depth 1 | 0 | 20 | 0 | 0.0% | 200 |

The combined baseline-relative proxy written by the evaluator is **176**.
The model made no illegal-move forfeits. It checkmated the random baseline in
3 games, but most random games reached the move cap. It lost every game to the
depth-1 Stockfish baseline. The reported proxy is a fixed toy comparison, not
an official FIDE or Lichess Elo, so this run does **not** support a claim of
600–700 Elo.

## Recommended presentation statement

> “This is a legal-move-constrained transformer policy/value model distilled
> from Stockfish. On the verified 30k-position checkpoint it learned useful
> move preferences and could beat a random mover, but it remains a research
> prototype: validation diverged from training and it was not competitive with
> even the depth-1 Stockfish baseline. The next improvement is the larger,
> multi-source 60k/200k-position curriculum run.”

## Generated evidence

- Full match report: `reports/evaluation.json`
- Evaluation log: `logs/evaluation.log.txt`
- Training log: `checkpoints/training.log.txt`
- Training summary: `checkpoints/training_summary.json`
