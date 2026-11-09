#!/bin/bash
# Replaces night.sh's final round + finale.sh. Waits for the expert round by PID, picks the best of
# baseline / DAgger / expert (blunder rate first, 1.5-point margin, then Elo), then rates that model with
# blended and material evaluation, since material beat learned on the baseline (1100 vs 990 Elo).
cd "$(dirname "$0")"
while kill -0 97348 2>/dev/null; do sleep 60; done
log() { echo "[$(TZ=America/New_York date +%H:%M)] morning: $*" | tee -a runs/selfplay.log; }
DEADLINE=$(TZ=America/New_York date -j -f "%Y-%m-%d %H:%M" "2026-09-18 08:00" +%s)
python3 - <<'PY'
import json, os
cands = [("baseline", "models/fused_bf16", "adapters_best", "runs/rating_baseline_learned.json"),
         ("sp_dagger", "models/sp_dagger", "runs/sp_dagger/adapters", "runs/sp_dagger/rating.json"),
         ("sp_expert", "models/sp_expert", "runs/sp_expert/adapters", "runs/sp_expert/rating.json")]
ok = []
for c in cands:
    try:
        ok.append((c, json.load(open(c[3]))))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
def better(x, y):
    if x["blunder_rate"] < y["blunder_rate"] - 0.015: return True
    if x["blunder_rate"] > y["blunder_rate"] + 0.015: return False
    return x["elo"] > y["elo"]
b = ok[0]
for c in ok[1:]:
    if better(c[1], b[1]): b = c
(name, model, adapter, _), r = b
json.dump({"name": name, "model": model, "adapter": adapter, "elo": r["elo"], "blunder_rate": r["blunder_rate"]},
          open("runs/best.json", "w"))
os.path.lexists("models/best") and os.remove("models/best")
os.symlink(os.path.basename(model), "models/best")
print(f"best: {name} ({model}) Elo {r['elo']} blunder {r['blunder_rate']}")
PY
log "$(python3 -c "import json;print('best:', json.load(open('runs/best.json')))")"
best=$(python3 -c "import json;print(json.load(open('runs/best.json'))['model'])")
for mode in blend material; do
  if (( $(date +%s) + 3900 < DEADLINE )); then
    log "rating $best with $mode evaluation"
    uv run rating.py --model "$best" --games 10 --sims 16 --value-mode $mode --out runs/final_$mode.json > runs/final_$mode.log 2>&1
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
