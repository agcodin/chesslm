#!/bin/bash
# After the blend rating: a second expert-iteration round from the new best (sp_expert), to test whether
# the gain repeats, then the morning summary. Waits on the blend rating's PID.
cd "$(dirname "$0")"
log() { echo "[$(TZ=America/New_York date +%H:%M)] dawn: $*" | tee -a runs/selfplay.log; }
while kill -0 98621 2>/dev/null; do sleep 30; done
log "blend: $(tail -1 runs/final_blend.log)"
log "expert iteration round 2 from sp_expert (2000 search-labelled positions)"
FROM_ADAPTER=runs/sp_expert/adapters FROM_MODEL=models/sp_expert SEED=101 MAXPOS=2000 \
  ./selfplay_round_from.sh search sp_night1 200 || log "round 2 failed"
python3 - <<'PY'
import json, os
try:
    b, c = json.load(open("runs/sp_expert/rating.json")), json.load(open("runs/sp_night1/rating.json"))
except (FileNotFoundError, json.JSONDecodeError):
    raise SystemExit
if c["blunder_rate"] < b["blunder_rate"] - 0.015 or (abs(c["blunder_rate"] - b["blunder_rate"]) <= 0.015 and c["elo"] > b["elo"]):
    json.dump({"name": "sp_night1", "model": "models/sp_night1", "adapter": "runs/sp_night1/adapters",
               "elo": c["elo"], "blunder_rate": c["blunder_rate"]}, open("runs/best.json", "w"))
    os.remove("models/best"); os.symlink("sp_night1", "models/best")
    print("round 2 PROMOTED")
else:
    print("round 2 not better; sp_expert stays best")
PY
python3 - <<'PY' | tee runs/morning_summary.txt
import json, glob
rows = []
for name, path in [("baseline, learned", "runs/rating_baseline_learned.json"),
                   ("baseline, material", "runs/rating_baseline_material.json"),
                   ("DAgger", "runs/sp_dagger/rating.json"), ("expert iteration", "runs/sp_expert/rating.json")] + \
                  [(p.split("/")[1], p) for p in sorted(glob.glob("runs/sp_night*/rating.json"))] + \
                  [("sp_expert, blend", "runs/final_blend.json")]:
    try:
        r = json.load(open(path))
        rows.append(f"{name:22} Elo {r['elo']:5} {str(r['elo_90ci']):14} blunder {r['blunder_rate']:.1%}  acpl {r['acpl']}  value={r['value']}")
    except Exception:
        rows.append(f"{name:22} (not run)")
print("\n".join(rows))
PY
log "summary written to runs/morning_summary.txt"
