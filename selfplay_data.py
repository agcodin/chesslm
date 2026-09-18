"""Training data from ChessLM's own games, so it learns the positions it actually reaches.

Supervised data comes from Stockfish self-play, but in real games ChessLM lands in positions that
never occur there (typically right after its own mistakes). This collects positions from ChessLM's
games -- half against itself, half against Stockfish skill 0 so it gets punished -- and labels them:

  --label stockfish  DAgger: Stockfish's best move (depth 12) and evaluation.
  --label search     Expert iteration (AlphaZero-style): ChessLM's own search choice as the move
                     target; value rows still use Stockfish, since game outcomes between weak
                     players are too noisy to learn evaluation from.

Output rows use the same formats as gen_data.py, so the two datasets can be mixed.
Usage: uv run selfplay_data.py --games 200 --label stockfish --out runs/dagger
"""
import argparse, json, math, os, random
import multiprocessing as mp
import chess, chess.engine
from common import bucket_letter, prompt, value_prompt
from engine import Policy, search


def pick(policy, board, rng, temperature):
    """Sample from the model's move distribution (temperature > 0) or take its top move."""
    pri = policy.priors(board)
    if temperature <= 0:
        return chess.Move.from_uci(max(pri, key=pri.get))
    moves = list(pri)
    w = [math.exp(pri[m] / temperature) for m in moves]
    return chess.Move.from_uci(rng.choices(moves, weights=w)[0])


def collect(policy, games, seed):
    """Return list of FENs where ChessLM was to move, from its own games."""
    rng = random.Random(seed)
    sf = chess.engine.SimpleEngine.popen_uci("stockfish")
    sf.configure({"Skill Level": 0})
    fens = []
    for g in range(games):
        board = chess.Board()
        vs_sf = g % 2 == 1
        me = chess.WHITE if g % 4 < 2 else chess.BLACK
        while not board.is_game_over(claim_draw=True) and board.ply() < 200:
            if vs_sf and board.turn != me:
                board.push(sf.play(board, chess.engine.Limit(time=0.02)).move)
                continue
            if board.ply() >= 4:
                fens.append(board.fen())
            # Sample early for opening variety, then play its real (greedy) moves.
            board.push(pick(policy, board, rng, 1.0 if board.ply() < 10 else 0.0))
        if (g + 1) % 20 == 0:
            print(f"{g + 1}/{games} games, {len(fens)} positions", flush=True)
    sf.quit()
    return fens


def sf_label(fens):
    eng = chess.engine.SimpleEngine.popen_uci("stockfish")
    eng.configure({"Threads": 1, "Hash": 32})
    out = []
    for fen in fens:
        board = chess.Board(fen)
        info = eng.analyse(board, chess.engine.Limit(depth=12))
        cp = info["score"].relative.score(mate_score=10000)
        out.append((fen, info["pv"][0].uci(), bucket_letter(cp)))
    eng.quit()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/fused_bf16")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--label", choices=["stockfish", "search"], required=True)
    ap.add_argument("--sims", type=int, default=12, help="search simulations for --label search")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-search-positions", type=int, default=4000,
                    help="--label search costs sims x 0.085s per position, so label a random sample")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    policy = Policy(a.model, None)
    fens = list(dict.fromkeys(collect(policy, a.games, a.seed)))  # dedupe, keep order
    if a.label == "search" and len(fens) > a.max_search_positions:
        fens = random.Random(a.seed).sample(fens, a.max_search_positions)
    print(f"{len(fens)} unique positions; labelling with Stockfish", flush=True)
    chunks = [fens[i::14] for i in range(14)]
    # spawn, not fork: forking a process that has already initialised Metal is unsafe on macOS.
    with mp.get_context("spawn").Pool(14) as pool:
        labelled = [row for part in pool.map(sf_label, chunks) for row in part]

    rows, agree = [], 0
    for i, (fen, sf_move, letter) in enumerate(labelled):
        board = chess.Board(fen)
        if a.label == "search":
            move = search(policy, board, sims=a.sims)[0].uci()
            if (i + 1) % 1000 == 0:
                print(f"searched {i + 1}/{len(labelled)}", flush=True)
        else:
            move = sf_move
        agree += move == sf_move
        rows.append({"prompt": prompt(board), "completion": " " + move})
        rows.append({"prompt": value_prompt(board), "completion": " " + letter})
    random.Random(a.seed).shuffle(rows)
    with open(f"{a.out}/train.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"wrote {len(rows)} rows to {a.out}/train.jsonl; move target matches Stockfish "
          f"{agree / max(len(labelled), 1):.1%} of the time")


if __name__ == "__main__":
    main()
