"""Label positions from randomized Stockfish self-play with Stockfish's best move AND its evaluation."""
import json, random, sys
from multiprocessing import Pool
import chess, chess.engine
from common import prompt, value_prompt, bucket_letter

def worker(seed, games=70):
    rng = random.Random(seed)
    eng = chess.engine.SimpleEngine.popen_uci("stockfish")
    eng.configure({"Threads": 1, "Hash": 32})
    rows = []
    for _ in range(games):
        board = chess.Board()
        while not board.is_game_over() and board.ply() < 160:
            info = eng.analyse(board, chess.engine.Limit(depth=12))
            best = info["pv"][0]
            if board.ply() >= 4:  # skip trivial opening duplicates
                rows.append({"prompt": prompt(board), "completion": " " + best.uci()})
                cp = info["score"].relative.score(mate_score=10000)
                rows.append({"prompt": value_prompt(board), "completion": " " + bucket_letter(cp)})
            # Some random moves so the data covers imperfect, human-like positions.
            board.push(rng.choice(list(board.legal_moves)) if rng.random() < 0.15 else best)
    eng.quit()
    return rows

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    seed0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out = sys.argv[3] if len(sys.argv) > 3 else "data"
    with Pool(n) as p:
        rows = [r for chunk in p.map(worker, range(seed0, seed0 + n)) for r in chunk]
    random.Random(0).shuffle(rows)
    k = len(rows) // 20
    for name, part in [("valid", rows[:k]), ("test", rows[k:2*k]), ("train", rows[2*k:])]:
        with open(f"{out}/{name}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in part)
    print(len(rows), "rows")
