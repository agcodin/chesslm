# ChessLM notes

## Setup
brew install stockfish uv; uv sync

## Pipeline
1. `uv run gen_data.py 12` → data/{train,valid,test}.jsonl (45,601 positions, ~1 min)
2. `uv run mlx_lm.lora --model Qwen/Qwen2.5-1.5B-Instruct --train --data data --iters 1500 --batch-size 8 --num-layers 16 --learning-rate 1e-4 --mask-prompt --adapter-path adapters`
3. `uv run play.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter adapters --games 10`

## Results log
| run | model | data | legal % | score vs SF skill 0 |
|---|---|---|---|---|
| baseline | Qwen2.5-0.5B, no training | – | 88.2% | 0/4 |

Top-1 agreement with Stockfish depth 12 on fixed test set (data/test.jsonl):
| run | top-1 |
|---|---|
| Qwen2.5-1.5B untrained | 6.5% (n=200) |
| 1.5B v1 long prompt, 1500 steps | ~8.5% (n=200, possibly contaminated by prompt change) |
| **1.5B v2 short prompt, 2000×32 — adapters_best** | **11.75% (n=400)** |
| 1.5B v2 @1000 steps | 9.75% |
| 3B-4bit v2 @1000 steps (3x slower to train) | 9.5% |
| improve round 1 (+91k positions, lr 5e-5) | 9.5% discarded |
| improve round 2 | 10.75% discarded |
Games vs SF skill 0: raw policy untrained 0.5/4, adapters_best 0/4; with PUCT search adapters_best 0/3 (sims 8 and 16), lasting 25-39 moves.
Takeaway: supervised move-matching plateaus ~10-12%; next gains need a learned value head / better search, not more of the same SFT.

## Publishing later (TODO)
- Ask for the GitHub repo name and personal site details (repo, framework, how it deploys)
- git init; .gitignore: data/ adapters/ runs/ .venv/ *.log
- Adapters too big for git → Hugging Face Hub or a GitHub release
- Site: project page + demo video/GIF + Elo chart

## Components (added 2026-09-17)
- engine.py: LLM priors over all legal moves (character-level trie walk + KV-cache trim) + PUCT search; leaf value = material/mobility heuristic (not learned yet)
- coach.py: RAG = exact-position opening lookup (Lichess chess-openings, 3,810 lines) + vector search (bge-small via fastembed) over kb/concepts.jsonl; base Qwen explains the move grounded in those
- server.py + web/index.html: `uv run uvicorn server:app --port 8000`
- eval_acc.py: top-1 agreement with Stockfish on the fixed data/test.jsonl
- improve.sh R1 R2: rounds of new data → resume LoRA from adapters_best → keep if top-1 improves; log in runs/improve.log

## Session 2026-09-17 morning
- v1 (prompt listed legal moves, ~450 tok): slow; top-1 vs Stockfish base 6.5% (200 pos) → trained ~8.5-10%
- v2: prompt = FEN only (legality enforced by trie), 211k positions in data_v2 (test set = original data/test.jsonl)
- overnight.sh (nohup, stops starting rounds so it ends by 2 PM EDT): 1.5B trial → 3B-4bit trial → winner → improve.sh rounds
- Logs: runs/overnight.log, runs/improve.log; best adapter: adapters_best/, model id in runs/best_model

## Web app
- launch: entry "chesslm" in ~/.claude/launch.json, or `uv run uvicorn server:app --port 8000`
- /api/move is async def on purpose: MLX streams are bound to the thread that loaded the model
- pieces: web/pieces = Lichess cburnett SVGs (CC BY-SA 3.0) — credit this when publishing

## Value head (session 2, 2026-09-17 pm)
- data_v3: every position labeled with Stockfish best move AND eval; eval encoded as one of 21 single letters a..u (tanh(cp/400) buckets) so one forward pass gives a distribution -> expected value. 180,770 rows.
- engine.Policy.value() + evaluate(board, policy); search(..., learned_value=True) uses it at leaves. play.py --material-value compares against the old handcrafted eval.
- Trained from the move-only adapter: val loss 3.80 (untrained on this task) -> 0.689 @500 -> 0.629 @1000 -> 0.600 @2000.
- Value quality @1000 steps (eval_value.py, n=150): corr 0.754 with Stockfish, MAE 0.388, sign agreement 78%.
- adapters_best now = move+value adapter (step 2000 of value training); adapters_move_only = previous 11.75% move-only adapter.
- NOT YET RUN: matches comparing learned value vs material value (play.py --games 4 --sims 16, with and without --material-value), and eval_acc on the new adapter to check the move skill did not regress.
- Caution: do NOT run eval scripts while training — doing so OOM'd the GPU and killed a run.

## Unattended rounds, evening 2026-09-17 (phase3.sh)
Gate: promote only if top-1 does not regress >0.005 AND top-1 or value corr improves.
| round | iters | lr | layers | batch | eval rows kept | top-1 | value corr | result |
|---|---|---|---|---|---|---|---|---|
| (baseline) move+value @2000 | | | | | 0.5 | 9.0% | 0.797 | starting point |
| 1 | 1200 | 1e-4 | 16 | 32 | 0.35 | 8.67% | 0.735 | discarded |
| 2 | 1500 | 1e-4 | 24 | 32 | 0.35 | – | – | OOM: 24 layers needs batch 16 |
| 3 | 1500 | 1e-4 | 16 | 32 | 0.20 | 12.33% | 0.769 | PROMOTED |
| 4 | 1200 | 5e-5 | 24 | 16 | 0.20 | 11.67% | 0.812 | discarded (best value-only model, kept at runs/p3r4/adapters) |
| 5 | 2000 | 7e-5 | 16 | 32 | 0.35 | **12.67%** | **0.867** | PROMOTED — current adapters_best |
Key lesson: a 50/50 move/value data mix cost move accuracy (11.75% -> 9.0%). Thinning value rows to
20-35% recovered it and beat the old move-only record, so one model can do both if the mix is right.
Two-specialist support exists (Policy(..., value_adapter=...), play.py --value-adapter) but round 5
beat both specialists, so it is currently unused.
Still 0/2 in games vs Stockfish skill 0 at sims=8.
Value ablation (2 games each, sims=16, vs SF skill 0): learned value 0/2, material value 0/2 — inconclusive.
Next: measure with finer signal than win/loss (game length, blunder rate by Stockfish eval drop per move,
or a weaker opponent such as SF depth 1 / a random mover) so learned-vs-material can actually be told apart.
