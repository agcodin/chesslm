#!/bin/bash
# Overnight driver: after the queued baseline/DAgger/expert runs finish, keep improving from the best
# model until 8:00 AM. Promotion uses blunder rate (measured over ~1200 moves, far less noisy than a
# 30-game rating): lower by >= 1.5 points wins; within that, the higher Elo wins.
set -uo pipefail
cd "$(dirname "$0")"
DEADLINE=$(TZ=America/New_York date -j -f "%Y-%m-%d %H:%M" "2026-09-18 08:00" +%s)
ROUND_SECS=7800   # ~2h10m: collect, label, fresh data, train, merge, evaluate, 30 rated games
log() { echo "[$(TZ=America/New_York date +%H:%M)] night: $*" | tee -a runs/selfplay.log; }

# The two methods under test, run directly. (An earlier queue waited with `pgrep -f <name>`, which
# matched its own command line and waited forever; running the steps in sequence avoids that.)
[ -f runs/sp_dagger/rating.json ] || ./selfplay_round.sh stockfish sp_dagger 200 || log "sp_dagger failed"
[ -f runs/sp_expert/rating.json ] || ./selfplay_round.sh search sp_expert 200 || log "sp_expert failed"
log "DAgger and expert-iteration rounds finished"

# candidates: name|model|adapter|rating file|label used to make it
best=$(python3 - <<'PY'
import json, os
cands = [("baseline", "models/fused_bf16", "adapters_best", "runs/rating_baseline_learned.json", "stockfish"),
         ("sp_dagger", "models/sp_dagger", "runs/sp_dagger/adapters", "runs/sp_dagger/rating.json", "stockfish"),
         ("sp_expert", "models/sp_expert", "runs/sp_expert/adapters", "runs/sp_expert/rating.json", "search")]
ok = []
for c in cands:
    try:
        r = json.load(open(c[3]))
        if os.path.isdir(c[1]) and os.path.isdir(c[2]):
            ok.append((c, r))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
def better(x, y):   # is rating x better than y
    if x["blunder_rate"] < y["blunder_rate"] - 0.015: return True
    if x["blunder_rate"] > y["blunder_rate"] + 0.015: return False
    return x["elo"] > y["elo"]
b = ok[0]
for c in ok[1:]:
    if better(c[1], b[1]): b = c
(name, model, adapter, rating, label), r = b
print(f"{name}|{model}|{adapter}|{rating}|{label}|{r['elo']}|{r['blunder_rate']}")
PY
)
IFS='|' read -r bname bmodel badapter brating blabel belo bblunder <<< "$best"
log "best so far: $bname (Elo $belo, blunder rate $bblunder); continuing with label=$blabel"

n=1
while (( $(date +%s) + ROUND_SECS < DEADLINE )); do
  name=sp_night$n
  log "round $n from $bname"
  FROM_ADAPTER=$badapter FROM_MODEL=$bmodel SEED=$((100 + n)) ./selfplay_round_from.sh "$blabel" "$name" 200 \
    || { log "round $n failed"; n=$((n + 1)); continue; }
  verdict=$(python3 - "$brating" "runs/$name/rating.json" <<'PY'
import json, sys
b, c = (json.load(open(p)) for p in sys.argv[1:3])
if c["blunder_rate"] < b["blunder_rate"] - 0.015 or (abs(c["blunder_rate"] - b["blunder_rate"]) <= 0.015 and c["elo"] > b["elo"]):
    print(f"PROMOTE|{c['elo']}|{c['blunder_rate']}")
else:
    print(f"KEEP|{c['elo']}|{c['blunder_rate']}")
PY
)
  IFS='|' read -r v elo blunder <<< "$verdict"
  if [[ $v == PROMOTE ]]; then
    bname=$name; bmodel=models/$name; badapter=runs/$name/adapters; brating=runs/$name/rating.json
    belo=$elo; bblunder=$blunder
    log "round $n PROMOTED: Elo $elo, blunder rate $blunder"
  else
    log "round $n not better (Elo $elo, blunder rate $blunder); best stays $bname"
  fi
  n=$((n + 1))
done

# Point the app at the winner.
ln -sfn "$(basename "$bmodel")" models/best
echo "{\"name\": \"$bname\", \"model\": \"$bmodel\", \"adapter\": \"$badapter\", \"elo\": $belo, \"blunder_rate\": $bblunder}" > runs/best.json
log "done. best = $bname (Elo $belo, blunder rate $bblunder); models/best -> $bmodel"
