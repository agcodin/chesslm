#!/bin/bash
# Fair size comparison at 1000 steps (32k examples each), then improvement rounds until 2 PM EDT.
set -uo pipefail
cd "$(dirname "$0")"
DEADLINE=$(TZ=America/New_York date -j -f "%Y-%m-%d %H:%M" "2026-09-17 14:00" +%s)
log() { echo "[$(TZ=America/New_York date +%H:%M)] $*" | tee -a runs/overnight.log; }
score() { MODEL="$1" uv run eval_acc.py "$2" 400 2>>runs/overnight.log | tail -1 | python3 -c "import json,sys;print(json.load(sys.stdin)['top1'])" 2>/dev/null || echo 0; }

until [ -f adapters_v2_3b/0001000_adapters.safetensors ]; do sleep 30; done
sleep 10; pkill -f "mlx_lm.lora --model mlx-community/Qwen2.5-3B"; sleep 10
cp adapters_v2_3b/0001000_adapters.safetensors adapters_v2_3b/adapters.safetensors
mkdir -p cmp15 && cp adapters_v2_1.5b/adapter_config.json cmp15/ && cp adapters_v2_1.5b/0001000_adapters.safetensors cmp15/adapters.safetensors
s3=$(score mlx-community/Qwen2.5-3B-Instruct-4bit adapters_v2_3b); log "3B-4bit step1000: top1=$s3"
s15=$(score Qwen/Qwen2.5-1.5B-Instruct cmp15); log "1.5B step1000: top1=$s15 (step2000 was 0.1175)"
if python3 -c "import sys; sys.exit(0 if float('$s3') > float('$s15') else 1)"; then
  WIN=mlx-community/Qwen2.5-3B-Instruct-4bit; rm -rf adapters_best; cp -r adapters_v2_3b adapters_best; echo $s3 > runs/best_score
else
  WIN=Qwen/Qwen2.5-1.5B-Instruct; rm -rf adapters_best; cp -r adapters_v2_1.5b adapters_best; echo 0.1175 > runs/best_score
fi
echo "$WIN" > runs/best_model; log "winner: $WIN"
# 3B rounds are ~3x slower; scale iterations so a round still fits ~45 min.
[[ $WIN == *3B* ]] && export ITERS=500
MODEL=$WIN DEADLINE=$DEADLINE ./improve.sh 1 20
log "done. best $(cat runs/best_score) with $WIN"
