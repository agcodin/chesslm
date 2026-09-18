# ChessLM

A small language model taught to play chess, trained entirely on a laptop.

ChessLM fine-tunes Qwen2.5-1.5B with LoRA on Apple Silicon (MLX), then plays through a
PUCT tree search in which the same model supplies both the move candidates and a learned
evaluation of each position. A retrieval-augmented coach explains its moves, and a web board
lets you play against it.

**Honest status:** it plays legal, recognisable chess but does not yet beat Stockfish at its
lowest setting. The numbers below are real measurements, including the experiments that failed.

## How it works

1. **Data.** Stockfish plays itself with 15% random moves mixed in, so positions look like
   imperfect play. Every position is labelled with Stockfish's best move (depth 12) and its
   evaluation.
2. **Two tasks, one model.** Each position becomes two training examples:
   `FEN: … Best move:` → `e2e4`, and `FEN: … Eval:` → one of 21 letters `a`–`u`
   (the evaluation squashed with `tanh(cp/400)` into buckets). Because every bucket is a single
   token, one forward pass gives a full probability distribution over evaluations, and its mean
   is a smooth value in [-1, 1].
3. **Legal moves only.** The model scores every legal move by walking a prefix tree over move
   characters and rewinding the KV cache between branches, so it can never play an illegal move.
4. **Search.** PUCT (the AlphaZero selection rule) expands the model's top moves and scores
   leaves with the learned evaluation.
5. **Coach.** Opening names come from an exact position lookup in the Lichess opening database.
   Chess principles are retrieved by vector search (bge-small embeddings) against facts computed
   from the move — captures, checks, forks, pins, hanging pieces — and the base model explains the
   move using only those facts.

## Results

Move accuracy is top-1 agreement with Stockfish on a fixed set of held-out positions. Value
correlation is between the model's evaluation and Stockfish's.

| Model | Move accuracy | Value correlation |
|---|---|---|
| Qwen2.5-1.5B, untrained | 6.5% | — |
| Moves only, 64k examples | 11.75% | — |
| Moves + value, 50/50 data mix | 9.0% | 0.797 |
| **Moves + value, value rows thinned to 35%** | **12.67%** | **0.867** |

What the experiments showed:

- **Prompt length mattered more than model size.** The first prompt listed every legal move
  (~450 tokens); training was so slow that one run saw only 28% of the data once. A
  position-only prompt (~60 tokens) fixed it — the search enforces legality anyway.
- **A bigger model was not better.** Qwen2.5-3B (4-bit) scored 9.5% against the 1.5B's 9.75%
  on identical data, while training about 3x slower.
- **Adding a value head initially hurt move choice.** A 50/50 mix of move and value examples
  dropped move accuracy from 11.75% to 9.0%. Keeping only 20–35% of the value examples recovered
  it and beat the move-only record, so one model can learn both if the mix is right.
- **Not yet winning.** Against Stockfish skill 0 the model scores 0 in every configuration tried,
  including learned vs. handcrafted evaluation (0/2 each at 16 simulations). Win/loss is too
  coarse to separate them; blunder rate or a weaker opponent is the next measurement.

## Running it

Requires an Apple Silicon Mac, [uv](https://docs.astral.sh/uv/) and Stockfish.

```bash
brew install stockfish uv
uv sync
```

Fetch the opening database for the coach (CC0, from lichess-org/chess-openings):

```bash
for f in a b c d e; do curl -sfL https://raw.githubusercontent.com/lichess-org/chess-openings/master/$f.tsv -o kb/$f.tsv; done
```

Generate data and train (about 2 minutes and 45 minutes respectively on an M5 Pro):

```bash
mkdir -p data_v3
uv run gen_data.py 14 500 data_v3
uv run mlx_lm.lora --model Qwen/Qwen2.5-1.5B-Instruct --train --data data_v3 --iters 2000 \
  --batch-size 32 --num-layers 16 --learning-rate 7e-5 --mask-prompt --adapter-path adapters_best
```

Evaluate and play:

```bash
uv run eval_acc.py adapters_best 300
uv run eval_value.py adapters_best 200
uv run play.py --games 2 --sims 16
uv run uvicorn server:app --port 8000
```

`phase3.sh` is the unattended loop used for the results above: it generates fresh data, trains
with varying hyperparameters and data mixes, and promotes a model only if move accuracy holds and
move or value quality improves. `NOTES.md` is the full lab notebook.

## Files

| File | Purpose |
|---|---|
| `gen_data.py` | Stockfish self-play → move and value training examples |
| `common.py` | Prompt formats and value buckets |
| `engine.py` | Move scoring over legal moves, learned value, PUCT search |
| `coach.py` | Opening lookup, move facts, vector retrieval, explanations |
| `server.py`, `web/` | FastAPI backend and the playable board |
| `eval_acc.py`, `eval_value.py`, `play.py` | Move accuracy, value correlation, games vs Stockfish |
| `phase3.sh` | Unattended train / evaluate / promote loop |

## Credits

Chess piece images: the cburnett set from [Lichess](https://github.com/lichess-org/lila),
licensed CC BY-SA 3.0. Opening names: [lichess-org/chess-openings](https://github.com/lichess-org/chess-openings) (CC0).
Base model: [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
