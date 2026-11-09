#!/bin/bash
# Full unattended plan until DEADLINE: finish 1.5B trial -> 3B-4bit trial -> pick winner -> improve rounds.
set -uo pipefail
cd "$(dirname "$0")"
DEADLINE=$(TZ=America/New_York date -j -f "%Y-%m-%d %H:%M" "2026-09-17 14:00" +%s)
log() { echo "[$(TZ=America/New_York date +%H:%M)] $*" | tee -a runs/overnight.log; }
mkdir -p runs
top1() { MODEL="$1" uv run eval_acc.py "$2" 400 2>>runs/overnight.log | tail -1; }

# 1) 1.5B trial: stop at step 2000
until [ -f adapters_v2_1.5b/0002000_adapters.safetensors ]; do sleep 30; done
sleep 10; pkill -f "mlx_lm.lora --model Qwen/Qwen2.5-1.5B-Instruct --train --data data_v2"; sleep 10
cp adapters_v2_1.5b/0002000_adapters.safetensors adapters_v2_1.5b/adapters.safetensors
r15=$(top1 Qwen/Qwen2.5-1.5B-Instruct adapters_v2_1.5b); log "1.5B step2000: $r15"

# 2) 3B 4-bit trial, same number of examples (2000 x 32)
uv run mlx_lm.lora --model mlx-community/Qwen2.5-3B-Instruct-4bit --train --data data_v2 --iters 2000 --batch-size 32 \
  --num-layers 16 --learning-rate 1e-4 --mask-prompt --adapter-path adapters_v2_3b --steps-per-eval 1000 \
  --save-every 1000 --steps-per-report 100 > train_v2_3b.log 2>&1 || log "3B training failed"
r3=$(top1 mlx-community/Qwen2.5-3B-Instruct-4bit adapters_v2_3b); log "3B-4bit step2000: $r3"

# 3) winner
s15=$(echo "$r15" | python3 -c "import json,sys;print(json.load(sys.stdin)['top1'])" 2>/dev/null || echo 0)
s3=$(echo "$r3" | python3 -c "import json,sys;print(json.load(sys.stdin)['top1'])" 2>/dev/null || echo 0)
if python3 -c "import sys; sys.exit(0 if float('$s3') > float('$s15') else 1)"; then
  WIN=mlx-community/Qwen2.5-3B-Instruct-4bit; rm -rf adapters_best; cp -r adapters_v2_3b adapters_best; echo $s3 > runs/best_score
else
  WIN=Qwen/Qwen2.5-1.5B-Instruct; rm -rf adapters_best; cp -r adapters_v2_1.5b adapters_best; echo $s15 > runs/best_score
fi
echo "$WIN" > runs/best_model; log "winner: $WIN"

# 4) improvement rounds until the deadline
MODEL=$WIN DEADLINE=$DEADLINE ./improve.sh 1 20
log "done. best $(cat runs/best_score) with $WIN"
