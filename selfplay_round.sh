#!/bin/bash
# One learn-from-own-games round: collect ChessLM's positions -> label -> mix with fresh Stockfish
# self-play data (so it doesn't forget) -> train from adapters_best -> merge -> measure -> rate.
# Usage: ./selfplay_round.sh <stockfish|search> <name> [games]
# Runs every GPU stage sequentially; concurrent GPU jobs OOM the Metal driver. Batch 16 (not 32):
# batch 32 peaks ~18.5GB and OOMed when other apps held memory; 2x steps keeps examples seen equal.
set -euo pipefail
cd "$(dirname "$0")"
label=$1; name=$2; games=${3:-200}
d=runs/$name; mkdir -p $d/data
log() { echo "[$(TZ=America/New_York date +%H:%M)] $name: $*" | tee -a runs/selfplay.log; }
BASE=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*/ | head -1)

log "collecting $games games, label=$label"
[ -s $d/own/train.jsonl ] && log "reusing collected positions" || uv run selfplay_data.py --games "$games" --label "$label" --out $d/own > $d/selfplay.log 2>&1
log "$(tail -1 $d/selfplay.log)"

log "fresh Stockfish self-play data"
sed -i '' 's/games=[0-9]*)/games=30)/' gen_data.py
uv run gen_data.py 14 $((9000 + RANDOM % 1000)) $d/data > $d/gen.log 2>&1

# Mix: every own-game row plus the fresh rows, then keep 35% of value rows (the ratio that worked).
python3 - "$d" <<'PY'
import json, random, sys
d = sys.argv[1]; rng = random.Random(0)
rows = [json.loads(l) for l in open(f"{d}/own/train.jsonl")] + [json.loads(l) for l in open(f"{d}/data/train.jsonl")]
rows = [r for r in rows if "\nEval:" not in r["prompt"] or rng.random() < 0.35]
rng.shuffle(rows)
open(f"{d}/data/train.jsonl", "w").writelines(json.dumps(r) + "\n" for r in rows)
print(len(rows), "training rows")
PY
cp data_v3/valid.jsonl $d/data/valid.jsonl; rm -f $d/data/test.jsonl

log "training"
uv run mlx_lm.lora --model "$BASE" --train --data $d/data --iters 3000 --batch-size 16 --num-layers 16 \
  --learning-rate 7e-5 --mask-prompt --resume-adapter-file adapters_best/adapters.safetensors \
  --adapter-path $d/adapters --steps-per-eval 100000 --save-every 3000 > $d/train.log 2>&1
uv run mlx_lm.fuse --model "$BASE" --adapter-path $d/adapters --save-path models/$name > /dev/null 2>&1

top1=$(MODEL=models/$name uv run eval_acc.py none 300 2>/dev/null | tail -1)
corr=$(MODEL=models/$name uv run eval_value.py none 200 2>/dev/null | tail -1)
log "move: $top1"
log "value: $corr"
uv run rating.py --model models/$name --games 10 --sims 16 --out $d/rating.json > $d/rating.log 2>&1
log "rating: $(tail -1 $d/rating.log)"
rm -rf $d/data
