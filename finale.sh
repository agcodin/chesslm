#!/bin/bash
# After night.sh exits: rate the winning model with material and blended evaluation (if time before
# 8:00), then write the morning summary. Waits on night.sh's PID -- not pgrep -f, which can match
# its own command line.
cd "$(dirname "$0")"
while kill -0 96078 2>/dev/null; do sleep 60; done
log() { echo "[$(TZ=America/New_York date +%H:%M)] finale: $*" | tee -a runs/selfplay.log; }
DEADLINE=$(TZ=America/New_York date -j -f "%Y-%m-%d %H:%M" "2026-09-18 08:00" +%s)
best=$(python3 -c "import json;print(json.load(open('runs/best.json'))['model'])" 2>/dev/null || echo models/fused_bf16)
log "best model: $best"
for mode in blend material; do
  if (( $(date +%s) + 4200 < DEADLINE )); then
    log "rating $best with $mode evaluation"
    uv run rating.py --model "$best" --games 10 --sims 16 --value-mode $mode \
      --out runs/final_$mode.json > runs/final_$mode.log 2>&1
    log "$mode: $(tail -1 runs/final_$mode.log)"
  else
    log "no time to rate $mode before 8:00"
  fi
done
python3 - <<'PY' | tee runs/morning_summary.txt
import json, glob
rows = []
for name, path in [("baseline, learned", "runs/rating_baseline_learned.json"),
                   ("baseline, material", "runs/rating_baseline_material.json"),
                   ("DAgger", "runs/sp_dagger/rating.json"), ("expert iteration", "runs/sp_expert/rating.json")] + \
                  [(p.split("/")[1], p) for p in sorted(glob.glob("runs/sp_night*/rating.json"))] + \
                  [("best, material", "runs/final_material.json"), ("best, blend", "runs/final_blend.json")]:
    try:
        r = json.load(open(path))
        rows.append(f"{name:22} Elo {r['elo']:5} {str(r['elo_90ci']):14} blunder {r['blunder_rate']:.1%}  acpl {r['acpl']}  value={r['value']}")
    except Exception:
        rows.append(f"{name:22} (not run)")
print("\n".join(rows))
PY
log "summary written to runs/morning_summary.txt"
