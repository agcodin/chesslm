"""Top-1 agreement with Stockfish on held-out positions (fixed test set, comparable across rounds)."""
import json, os, sys
import chess
from engine import Policy

adapter = sys.argv[1] if len(sys.argv) > 1 else "adapters"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
p = Policy(os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct"), adapter=adapter if adapter != "none" else None)
rows = [json.loads(l) for l in open("data/test.jsonl")][:n]
hit = 0
for r in rows:
    board = chess.Board(r["prompt"].split("\n")[0][5:])
    hit += p.best(board).uci() == r["completion"].strip()
print(json.dumps({"adapter": adapter, "n": n, "top1": round(hit / n, 4)}))
