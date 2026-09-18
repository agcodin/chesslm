# ChessLM

A small language model taught to play chess, trained entirely on a laptop.

**Project page, with a replay of a real game and its search visible:** https://agcodin.github.io/chesslm/

**Play it on an Apple Silicon Mac:** `git clone https://github.com/agcodin/chesslm && cd chesslm && ./play.sh`

ChessLM fine-tunes Qwen2.5-1.5B with LoRA on Apple Silicon (MLX), then plays through a
PUCT tree search in which the same model supplies both the move candidates and a learned
evaluation of each position. A retrieval-augmented coach explains its moves, and a web board
lets you play against it.

**Headline:** learning from its own games cut ChessLM's blunder rate from 32% to about 27% of moves
(replicated), and it plays at roughly **1000 Elo** on a ladder anchored to Stockfish's calibrated
1320 setting. It does not yet beat that setting. Every number below is a real measurement, including
the experiments that failed and one prediction of mine that turned out wrong.

## How it works

1. **Data.** Stockfish plays itself with 15% random moves mixed in, so positions look like
   imperfect play. Every position is labelled with Stockfish's best move (depth 12) and its
   evaluation.
2. **Two tasks, one model.** Each position becomes two training examples:
   `FEN: … Best move:` → `e2e4`, and `FEN: … Eval:` → one of 21 letters `a`–`u`
   (the evaluation squashed with `tanh(cp/400)` into buckets). Because every bucket is a single
   token, one forward pass gives a full probability distribution over evaluations, and its mean
   is a smooth value in [-1, 1].
3. **Legal moves only, in one batch.** The model scores every legal move exactly: the position's KV
   cache is computed once and copied per candidate, so all moves are scored in one batched pass
   (13x faster than walking a prefix tree over move characters). It can never play an illegal move.
4. **Search.** PUCT (the AlphaZero selection rule) expands the model's top moves and scores
   leaves with the learned evaluation.
5. **Learning from its own games.** ChessLM plays 200 games; its search's choice in each position
   becomes the new training target (expert iteration, the AlphaZero idea), mixed with fresh
   Stockfish data so it doesn't forget.
6. **Coach.** Opening names come from an exact position lookup in the Lichess opening database.
   Chess principles are retrieved by vector search (bge-small embeddings) against facts computed
   from the move — captures, checks, forks, pins, hanging pieces — and the base model explains the
   move using only those facts.

## Results

### Playing strength

Rated on a ladder: a random mover, Stockfish limited to a one-ply search, Stockfish skill 0 and
Stockfish's calibrated `UCI_Elo 1320`, fitted jointly (Bradley-Terry) with a bootstrap 90% interval.
Blunder rate is the share of ChessLM's moves that lose at least 300 centipawns against Stockfish's
best move; over ~1,200 moves per run it is far more precise than a 30-game rating.

| Model (16 search simulations) | Elo [90% CI] | Blunder rate | Avg centipawn loss |
|---|---|---|---|
| Supervised on Stockfish data | 990 [851, 1147] | 32.1% | 297 |
| + DAgger (own positions, Stockfish labels) | 949 [823, 1091] | 30.6% | 279 |
| **+ expert iteration (own positions, own search labels)** | **1023 [888, 1181]** | **26.3%** | **240** |
| same model, independent re-rating | 1023 [888, 1181] | 27.3% | 249 |
| expert iteration, 2nd round | 990 [840, 1150] | 28.1% | 253 |

The ~5-point blunder reduction is several standard errors and replicated; the Elo differences are
within their intervals.

### Move and value accuracy

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
- **Copying its own search beat copying Stockfish, and I predicted the opposite.** ChessLM's search
  agreed with Stockfish in only 17% of its own positions, so I expected training on those choices
  to make it worse. Move accuracy did fall (13.3% to 11.3%), but blunders dropped sharply: the
  search's disagreements are mostly rejections of moves its lookahead shows to be losing. At this
  level, not blundering matters far more than finding the exact best move, and move accuracy was
  the wrong metric to judge it by.
- **Better evaluation is not automatically better search.** Before expert iteration, a plain
  material count beat the learned evaluation inside the search (29.0% vs 32.1% blunders). After
  it, the learned evaluation won clearly (26.3% vs 33.5% for a learned/material blend), plausibly
  because the policy was trained to agree with a search that used it.
- **A second round did not add more** (28.1%, within noise of round 1, with half the positions).
- **Compression did not speed it up.** 8-bit and 4-bit versions were no faster at 1.5B parameters
  (the model is not memory-bandwidth-bound at this size) and 4-bit lost accuracy.
- **Not yet winning against Stockfish's 1320 setting**: one draw in 30 games across all versions.

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

Merge the adapter (full precision is fastest; see results) and point `models/best` at it:

```bash
uv run mlx_lm.fuse --model Qwen/Qwen2.5-1.5B-Instruct --adapter-path adapters_best --save-path models/fused_bf16
ln -sfn fused_bf16 models/best
```

Learn from its own games, rate, and play:

```bash
./selfplay_round.sh search sp_expert 200
uv run rating.py --model models/sp_expert --games 10 --sims 16
uv run uvicorn server:app --port 8000
```

`phase3.sh` is the unattended loop used for the supervised results: it generates fresh data, trains
with varying hyperparameters and data mixes, and promotes a model only if move accuracy holds and
move or value quality improves. `NOTES.md` is the full lab notebook.

## Files

| File | Purpose |
|---|---|
| `gen_data.py` | Stockfish self-play → move and value training examples |
| `common.py` | Prompt formats and value buckets |
| `engine.py` | Batched move scoring, learned value, PUCT search |
| `selfplay_data.py`, `selfplay_round.sh` | Positions from its own games, labelled by Stockfish or its own search; one full train-and-rate round |
| `rating.py` | Elo ladder, bootstrap interval, blunder rate |
| `play.sh`, `weights/` | One-command local launcher; the trained 21 MB LoRA adapter |
| `record_game.py`, `docs/` | Annotated game recorder; the project page (`python3 docs/build.py`), served by GitHub Pages |
| `coach.py` | Opening lookup, move facts, vector retrieval, explanations |
| `server.py`, `web/` | FastAPI backend and the playable board |
| `eval_acc.py`, `eval_value.py`, `play.py` | Move accuracy, value correlation, games vs Stockfish |
| `phase3.sh` | Unattended train / evaluate / promote loop |

## Credits

Chess piece images: the cburnett set from [Lichess](https://github.com/lichess-org/lila),
licensed CC BY-SA 3.0. Opening names: [lichess-org/chess-openings](https://github.com/lichess-org/chess-openings) (CC0).
Base model: [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
