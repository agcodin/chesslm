"""Seconds per move-scoring call and per value call on fixed positions, for a given model."""
import json, random, sys, time
import chess
from engine import Policy

model = sys.argv[1]
adapter = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "none" else None
p = Policy(model, adapter)
rng = random.Random(3); boards = []
while len(boards) < 20:
    b = chess.Board()
    for _ in range(rng.randrange(4, 60)):
        if b.is_game_over(): break
        b.push(rng.choice(list(b.legal_moves)))
    if not b.is_game_over(): boards.append(b)
p.priors(boards[0]); p.value(boards[0])  # warm up
t = time.time(); [p.priors(b) for b in boards]; tp = (time.time() - t) / len(boards)
t = time.time(); [p.value(b) for b in boards]; tv = (time.time() - t) / len(boards)
print(json.dumps({"model": model, "priors_s": round(tp, 4), "value_s": round(tv, 4)}))
