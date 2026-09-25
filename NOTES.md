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

## Publishing (done 2026-09-18)
- Repo: https://github.com/agcodin/chesslm (public). Project page: https://agcodin.github.io/chesslm/
- Still open: adapters are in weights/ as a 21 MB LoRA; larger artefacts would need HF Hub or a release.

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

## Speed (#4) and rating (#1), night of 2026-09-17
- priors() now batched: prompt KV cache copied per legal move, one pass per move length. 13x faster than the trie
  (1.15s -> 0.087s/position), same top-3 moves in float32 (bf16 rounding explains the small diffs).
- Merged adapter into models/fused_bf16. 8-bit and 4-bit were NOT faster (1.5B is not bandwidth-bound) and 4-bit lost
  move accuracy (11.67% vs 13.33%), so full precision is the default. Per search step: ~0.085s (was ~1.2s).
- rating.py: Elo ladder anchored to Stockfish UCI_Elo 1320 (random mover, SF depth 1, SF skill 0 @0.05s, SF 1320 @0.1s),
  joint Bradley-Terry fit with one virtual draw per pairing, stratified bootstrap 90% interval, blunder rate (>=300cp loss).
- Baseline (fused_bf16, learned value, sims 16): Elo 990 [851, 1147]; blunder rate 32.2%; ACPL 297;
  vs random 6.5/10, vs SF depth1 0.5/10, vs SF skill0 0/10. Random mover's placement (906) is only bounded (it lost
  every game), which biases ChessLM's estimate upward; treat as ~900-1000.
- Value ablation, same ladder (fused_bf16, sims 16): MATERIAL eval Elo 1100 [947, 1258], blunder 29.0%, ACPL 265 vs
  LEARNED eval Elo 990 [851, 1147], blunder 32.2%, ACPL 297. ~1.7 SE on blunder rate: suggestive, all metrics agree.
  The learned value (corr 0.867 with SF) does not yet help search; likely static-eval error mid-exchange.
  Added search(value_mode="blend") = mean of learned and material, and rating.py --value-mode; not yet rated.

## Learning from its own games (#2), night of 2026-09-18
- selfplay_data.py: ChessLM plays 200 games (half vs itself, half vs SF skill 0; sampled first 10 plies, then greedy).
  selfplay_round.sh mixes those positions with fresh SF self-play data, keeps 35% of value rows, trains 3000 x 16.
  (Batch 32 OOMed once other apps held unified memory; batch 16 peaks at 10.9GB.)
- DAgger (own positions, Stockfish labels; 11,911 positions): top-1 13.33% (= baseline), value corr 0.779 (down from
  0.867 on the SF-self-play test set, which no longer matches its training distribution).
  Rating (learned value, sims 16): Elo 949 [823, 1091]; blunder 30.55% (baseline 32.15%); ACPL 279 (baseline 297);
  vs random 7.5/10. Modest real gain in move quality; Elo change within noise (random anchor moved 906 -> 805).
- Expert iteration (own positions, ChessLM search labels at sims 12, 4,000-position sample): search agreed with
  Stockfish on only 16.6% of these positions. Offline: top-1 11.33% (down), value corr 0.842.
  Rating: Elo 1023 [888, 1181]; BLUNDER 26.3% (baseline 32.2%, DAgger 30.6%); ACPL 240 (baseline 297).
  ~6-point blunder drop is >3 SE: real. I predicted this would hurt and was wrong: the search's disagreements with
  Stockfish are mostly rejections of moves its lookahead shows losing, so the policy learned to avoid blunders at
  the cost of exact best-move agreement. Top-1 vs Stockfish is the wrong metric for this; blunder rate is right.
  Promoted: models/best -> sp_expert.
- Queue bug: a waiter using `pgrep -f name` matched its own command line and waited forever (lost ~30 min).
  Drivers now wait on PIDs.
- Blend eval on sp_expert (mean of learned + material): Elo 1073 [909, 1235] but blunder 33.5%, ACPL 280 (learned on
  the same model: 26.3%, 240). Conflicting; blunder gap is large, Elo gap is noise -> keep learned. Plausible reason:
  expert iteration trained the policy to agree with a learned-value search, so swapping the eval breaks that match.
  The baseline's "material beats learned" therefore does not transfer to the expert-iteration model.
- Expert iteration round 2 (from sp_expert, 2,000 search-labelled positions, search agreed with SF 13.0%):
  top-1 12.0%, corr 0.827; Elo 990 [840, 1150], blunder 28.1%, ACPL 253. Not better than round 1 (26.3%);
  ~1 SE, and half the labelled positions. sp_expert stays best (models/best -> sp_expert).
- Replication rating of sp_expert running (runs/sp_expert/rating_repeat.json) to check the 26.3% headline.
- Replication of sp_expert (independent games: 1,275 moves vs 1,324): blunder 27.3%, ACPL 249 (first run 26.3%, 240).
  Pooled ~26.8% vs baseline 32.1%. Headline holds. play.py / server.py / rating.py now default to models/best.


## Compression study (session 2026-09-25)
Question: does compression damage a narrow skill faster than perplexity admits? Chess gives a
ground truth that fluency cannot fake, so eval_compress.py scores four axes at once -- perplexity,
unconstrained legal-move rate, Stockfish top-1, value correlation.

- compress.py factorises the three MLP projections (75% of params). Truncation goes through the
  Gram matrix of the narrow (1536) side: a direct SVD of an 8960-row projection asks for an
  8960x8960 U and GPU-times-out. Verified against numpy's SVD to 5 significant figures.
- mlx_lm.fuse CLI is unusable offline (wants a complete Hub snapshot incl. README/LICENSE);
  compress.load_fused does the same merge in memory.
- Findings at n_move=120 (runs/compress/results.jsonl, table.md):
  - quantisation is nearly free: 8-bit lossless, 4-bit ~1% perplexity for ~10% relative skill.
  - plain SVD is catastrophic far earlier than expected: 7% smaller -> ppl 3.2x, legality 65% -> 0%.
  - activation-aware (ASVD, calibrated on chess positions) is worth 1-2 orders of magnitude:
    at 23% smaller, ppl 280.6 -> 9.8.
  - THE RESULT: asvd keep=0.8 (15% smaller) has ppl 1.83x and value corr 87% retained, while
    legality falls 65% -> 19% with separated Wilson intervals. Two metrics say fine, model is broken.
- top1 is too noisy to carry the claim: baseline only 15.5%, trie floors a dead model at 4-6%,
  interval ~+-5pp at n=120. Legality does the work. Do not headline top1.
- Healing at 400 iters x batch 4 FAILED to recover (ppl 9.84 -> 8.70, skill flat). 1,600 examples
  vs the 64,000 the adapter saw. Long runs (3000 x 8) as stage3; heal.py folds LoRA back into the
  factors so size is unchanged.
- CARE: do not compare heal.py's training loss to the "val loss 0.600" logged above. That 0.600 is
  value-task validation loss; heal.py trains on the mixed move+value data and reports a running
  train loss. Different data, different split -- not a like-for-like floor. To state a real
  reference, measure the DENSE model's loss with heal.loss_fn on the same batches. Not done yet.
- CAUTION confirmed again: never run an eval alongside training. Stages are separate processes.
