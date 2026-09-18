#!/bin/bash
# Unattended improvement until DEADLINE. Everything runs SEQUENTIALLY: concurrent GPU work OOMs the Metal driver.
# Each round: fresh data -> train (varying hyperparameters) -> move test -> value test -> match vs Stockfish.
# A candidate is promoted only if it does not regress move skill and improves value or move quality.
set -uo pipefail
cd "$(dirname "$0")"
MODEL=Qwen/Qwen2.5-1.5B-Instruct
DEADLINE=$(TZ=America/New_York date -j -f "%Y-%m-%d %H:%M" "2026-09-17 20:30" +%s)
mkdir -p runs
log() { echo "[$(TZ=America/New_York date +%H:%M)] $*" | tee -a runs/phase3.log; }
jget() { python3 -c "import json,sys;print(json.load(sys.stdin).get('$1',0))" 2>/dev/null || echo 0; }
enough() { (( $(date +%s) + $1 < DEADLINE )); }

# Hyperparameter variations, tried in order; "adjust the method" rather than repeating one recipe.
# iters lr layers eval_fraction. Mixed 50/50 data cost move accuracy (11.75%->9.0%), so most
# rounds keep only a fraction of the rating rows and let the move task dominate again.
# iters lr layers eval_fraction batch. 24 layers at batch 32 OOMs on 24GB, so those rounds use batch 16.
CFG=("1200 5e-5 16 0.2 32" "1500 1e-4 24 0.35 16" "1500 1e-4 16 0.2 32" "1200 5e-5 24 0.2 16" "2000 7e-5 16 0.35 32")

best_top1=$(cat runs/best_top1 2>/dev/null || echo 0.1175)
best_corr=$(cat runs/best_corr 2>/dev/null || echo 0.754)
log "start: best_top1=$best_top1 best_corr=$best_corr deadline 20:30"

for r in $(seq ${START_ROUND:-1} 12); do
  enough 2700 || { log "not enough time for another round, stopping"; break; }
  set -- ${CFG[$(( (r - 1) % ${#CFG[@]} ))]}
  iters=$1; lr=$2; layers=$3; frac=$4; bs=$5
  d=runs/p3r$r; mkdir -p $d/data
  log "round $r: iters=$iters lr=$lr layers=$layers eval_rows=$frac batch=$bs"

  uv run gen_data.py 14 $((5000 + r * 100)) $d/data >> runs/phase3.log 2>&1 || { log "round $r: data generation failed"; continue; }
  cp data_v3/valid.jsonl $d/data/valid.jsonl; rm -f $d/data/test.jsonl
  python3 - "$d/data/train.jsonl" "$frac" <<'PY'
import json, random, sys
path, frac = sys.argv[1], float(sys.argv[2])
rng = random.Random(0)
rows = [json.loads(l) for l in open(path)]
kept = [r for r in rows if "\nEval:" not in r["prompt"] or rng.random() < frac]
open(path, "w").writelines(json.dumps(r) + "\n" for r in kept)
print(f"{len(rows)} rows -> {len(kept)} after thinning eval rows to {frac}")
PY

  uv run mlx_lm.lora --model $MODEL --train --data $d/data --iters $iters --batch-size $bs \
    --num-layers $layers --learning-rate $lr --mask-prompt \
    --resume-adapter-file adapters_best/adapters.safetensors --adapter-path $d/adapters \
    --steps-per-eval 100000 --save-every $iters --steps-per-report 200 > $d/train.log 2>&1 \
    || { log "round $r: training failed (see $d/train.log)"; continue; }

  top1=$(uv run eval_acc.py $d/adapters 300 2>>runs/phase3.log | tail -1 | jget top1)
  corr=$(uv run eval_value.py $d/adapters 200 2>>runs/phase3.log | tail -1 | jget corr)
  log "round $r: top1=$top1 corr=$corr"
  echo "{\"round\":$r,\"iters\":$iters,\"lr\":\"$lr\",\"layers\":$layers,\"frac\":$frac,\"batch\":$bs,\"top1\":$top1,\"corr\":$corr}" >> runs/metrics.jsonl

  # Promote only if move skill holds up and something actually improved.
  if python3 -c "
import sys
t,c,bt,bc = float('$top1'), float('$corr'), float('$best_top1'), float('$best_corr')
sys.exit(0 if t >= bt - 0.005 and (t > bt or c > bc) else 1)"; then
    rm -rf adapters_best_prev && cp -r adapters_best adapters_best_prev
    rm -rf adapters_best && cp -r $d/adapters adapters_best
    best_top1=$top1; best_corr=$corr
    echo $best_top1 > runs/best_top1; echo $best_corr > runs/best_corr
    log "round $r PROMOTED (top1=$top1 corr=$corr)"
    if enough 1500; then
      s=$(uv run play.py --games 2 --sims 8 --adapter adapters_best 2>/dev/null | tail -1)
      log "round $r match: $s"
    fi
  else
    log "round $r discarded (best top1=$best_top1 corr=$best_corr)"
  fi
  rm -rf $d/data
done
log "finished. best_top1=$best_top1 best_corr=$best_corr"
