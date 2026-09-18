"""How well does the learned evaluation match Stockfish on held-out positions?"""
import json, os, sys
import chess
import numpy as np
from engine import Policy
from common import letter_values

adapter = sys.argv[1] if len(sys.argv) > 1 else "adapters_v3"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
p = Policy(os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct"), adapter if adapter != "none" else None)
vals = letter_values()
rows = [json.loads(l) for l in open("data_v3/test.jsonl") if "\nEval:" in json.loads(l)["prompt"]][:n]
pred, true = [], []
for r in rows:
    board = chess.Board(r["prompt"].split("\n")[0][5:])
    pred.append(p.value(board))
    true.append(vals[ord(r["completion"].strip()) - ord("a")])
pred, true = np.array(pred), np.array(true)
print(json.dumps({"adapter": adapter, "n": len(rows),
                  "corr": round(float(np.corrcoef(pred, true)[0, 1]), 3),
                  "mae": round(float(np.abs(pred - true).mean()), 3),
                  "sign_agree": round(float(((pred > 0) == (true > 0)).mean()), 3)}))
