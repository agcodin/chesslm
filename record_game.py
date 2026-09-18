"""Record annotated games for the project page's replay viewer.

For every ply: the position after it and the move. For ChessLM's moves also: its top search candidates
with visit counts, its own evaluation, the plain-language move facts the coach uses, and Stockfish's
verdict (centipawns lost versus its best move, depth 12). Plays several games against Stockfish skill 0
and writes them all; the page picks one.

Usage: uv run record_game.py --games 4 --sims 16 --out runs/showcase_games.json
"""
import argparse, json
import chess, chess.engine
from engine import Policy, search
from coach import features, load_openings

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/best")
ap.add_argument("--games", type=int, default=4)
ap.add_argument("--sims", type=int, default=16)
ap.add_argument("--out", default="runs/showcase_games.json")
a = ap.parse_args()

policy = Policy(a.model, None)
book = load_openings()
opp = chess.engine.SimpleEngine.popen_uci("stockfish")
opp.configure({"Skill Level": 0})
judge = chess.engine.SimpleEngine.popen_uci("stockfish")


def cp(board, color):
    return judge.analyse(board, chess.engine.Limit(depth=12))["score"].pov(color).score(mate_score=3000)


games = []
for g in range(a.games):
    me = chess.WHITE if g % 2 == 0 else chess.BLACK
    board, plies, opening = chess.Board(), [], None
    while not board.is_game_over(claim_draw=True) and board.ply() < 200:
        if board.turn == me:
            best_cp = cp(board, me)
            move, visits = search(policy, board, sims=a.sims)
            top = sorted(visits.items(), key=lambda kv: -kv[1])[:4]
            ply = {"by": "chesslm", "san": board.san(move), "uci": move.uci(),
                   "value": round(policy.value(board), 2),
                   "candidates": [{"san": board.san(m), "visits": n} for m, n in top],
                   "facts": [f for f in features(board, move) if not f.startswith("material balance")]}
            board.push(move)
            ply["loss"] = max(0, min(1000, best_cp - cp(board, me)))
        else:
            move = opp.play(board, chess.engine.Limit(time=0.05)).move
            ply = {"by": "stockfish", "san": board.san(move), "uci": move.uci()}
            board.push(move)
        ply["fen"] = board.fen()
        opening = book.get(board.epd(), opening)
        plies.append(ply)
    mine = [p for p in plies if p["by"] == "chesslm"]
    games.append({"chesslm": "white" if me == chess.WHITE else "black", "result": board.result(claim_draw=True),
                  "opening": opening, "plies": plies,
                  "blunders": sum(p["loss"] >= 300 for p in mine), "moves": len(mine)})
    print(f"game {g + 1}: ChessLM {games[-1]['chesslm']}, {games[-1]['result']}, {len(plies)} plies, "
          f"{games[-1]['blunders']}/{len(mine)} blunders, opening {opening}", flush=True)

opp.quit(); judge.quit()
json.dump(games, open(a.out, "w"))
print("wrote", a.out)
