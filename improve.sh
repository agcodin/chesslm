#!/bin/bash
# Self-improvement rounds: fresh Stockfish-labeled positions -> continue LoRA from adapters_best -> keep only if
# top-1 agreement on the fixed data/test.jsonl improves. Stops starting new rounds after DEADLINE (epoch secs).
# Usage: MODEL=... DEADLINE=... ./improve.sh <first_round> <last_round>
set -uo pipefail
cd "$(dirname "$0")"
MODEL=${MODEL:?set MODEL}; DEADLINE=${DEADLINE:-9999999999}; ITERS=${ITERS:-1500}
mkdir -p runs
best=$(cat runs/best_score 2>/dev/null || echo 0)
for r in $(seq "$1" "$2"); do
  # A round takes ~45 min; don't start one that can't finish before the deadline.
  if (( $(date +%s) + 2700 > DEADLINE )); then echo "deadline reached, stopping" | tee -a runs/improve.log; break; fi
  d=runs/r$r; mkdir -p $d/data
  echo "== round $r $(date) best=$best" | tee -a runs/improve.log
  sed -i '' 's/games=[0-9]*)/games=70)/' gen_data.py
  uv run gen_data.py 14 $((r * 1000)) $d/data >> runs/improve.log 2>&1
  cp data_v2/valid.jsonl $d/data/valid.jsonl; rm -f $d/data/test.jsonl
  uv run mlx_lm.lora --model "$MODEL" --train --data $d/data --iters $ITERS --batch-size 32 \
    --num-layers 16 --learning-rate 5e-5 --mask-prompt --resume-adapter-file adapters_best/adapters.safetensors \
    --adapter-path $d/adapters --steps-per-eval 100000 --save-every $ITERS > $d/train.log 2>&1 \
    || { echo "round $r train failed" | tee -a runs/improve.log; continue; }
  score=$(MODEL="$MODEL" uv run eval_acc.py $d/adapters 400 2>>runs/improve.log | tail -1 | tee -a runs/improve.log \
    | python3 -c "import json,sys;print(json.load(sys.stdin)['top1'])")
  if python3 -c "import sys; sys.exit(0 if float('$score') > float('$best') else 1)"; then
    rm -rf adapters_best && cp -r $d/adapters adapters_best && best=$score && echo $best > runs/best_score
    echo "round $r KEPT top1=$score" | tee -a runs/improve.log
  else
    echo "round $r discarded top1=$score (best $best)" | tee -a runs/improve.log
  fi
  rm -rf $d/data
done
