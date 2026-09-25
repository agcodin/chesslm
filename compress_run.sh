#!/bin/bash
# Reproduce the compression study end to end. Roughly 2-3 hours on an M-series Mac.
#
# Stages are separate processes on purpose: MLX holds GPU memory for the life of a process, and
# running an eval alongside training has OOM'd this machine before. Nothing here runs in parallel.
set -euo pipefail
cd "$(dirname "$0")"

ADAPTER=${ADAPTER:-adapters_best}
N_MOVE=${N_MOVE:-120}
N_VALUE=${N_VALUE:-100}
CONFIRM_N=${CONFIRM_N:-800}
mkdir -p runs/compress

echo "== 1/6 uncompressed baseline =="
uv run python eval_compress.py Qwen/Qwen2.5-1.5B-Instruct --adapter "$ADAPTER" \
  --n-move "$N_MOVE" --n-value "$N_VALUE" --tag baseline_fp16 --out runs/compress/results.jsonl

echo "== 2/6 sweep: quantisation, plain SVD, activation-aware SVD =="
uv run python sweep.py --adapter "$ADAPTER" --arms quant,svd,asvd \
  --bits 8,4,3 --keeps 0.9,0.8,0.7,0.6,0.5,0.4,0.3 --n-move "$N_MOVE" --n-value "$N_VALUE"

echo "== 3/6 per-layer sensitivity profile =="
uv run python sensitivity.py profile --adapter "$ADAPTER" --arm asvd --keep 0.3 --n-ppl 60

echo "== 4/6 healing =="
for KEEP in 0.7 0.5; do
  uv run python heal.py --arm asvd --keep "$KEEP" --adapter "$ADAPTER" --iters 400
  uv run python eval_compress.py "models/healed_asvd_keep$KEEP" \
    --n-move "$N_MOVE" --n-value "$N_VALUE" --tag "healed_keep$KEEP" \
    --out runs/compress/results.jsonl
done

echo "== 5/6 high-n confirmation of the headline claim =="
uv run python eval_compress.py Qwen/Qwen2.5-1.5B-Instruct --adapter "$ADAPTER" \
  --n-move "$CONFIRM_N" --n-value 400 --tag baseline_fp16 --out runs/compress/confirm.jsonl
uv run python sweep.py --adapter "$ADAPTER" --arms quant --bits 4,3 \
  --n-move "$CONFIRM_N" --n-value 400 --out runs/compress/confirm.jsonl

echo "== 6/6 figures and tables =="
uv run python figures.py
uv run python report.py
uv run python report.py --results runs/compress/confirm.jsonl --out runs/compress/table_confirm.md
echo "done -- see runs/compress/ and docs/figures/"
