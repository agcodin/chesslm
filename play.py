"""Play ChessLM against Stockfish at a fixed skill level. --sims 0 = raw policy, >0 = PUCT search."""
import argparse
import chess, chess.engine
from engine import Policy, search

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/best", help="merged model; models/best links to the current best")
ap.add_argument("--adapter", default="none")
ap.add_argument("--games", type=int, default=4)
ap.add_argument("--skill", type=int, default=0)
ap.add_argument("--sims", type=int, default=0)
ap.add_argument("--value-adapter", default=None, help="separate adapter for evaluation only")
ap.add_argument("--material-value", action="store_true", help="use the handcrafted eval instead of the learned one")
a = ap.parse_args()

policy = Policy(a.model, a.adapter if a.adapter != "none" else None, a.value_adapter)
sf = chess.engine.SimpleEngine.popen_uci("stockfish")
sf.configure({"Skill Level": a.skill})
score = 0.0
for g in range(a.games):
    board, me = chess.Board(), chess.WHITE if g % 2 == 0 else chess.BLACK
    while not board.is_game_over(claim_draw=True) and board.ply() < 200:
        if board.turn == me:
            move = search(policy, board, sims=a.sims, learned_value=not a.material_value)[0] if a.sims else policy.best(board)
        else:
            move = sf.play(board, chess.engine.Limit(time=0.05)).move
        board.push(move)
    r = board.result(claim_draw=True)
    s = 0.5 if r in ("1/2-1/2", "*") else float((r == "1-0") == (me == chess.WHITE))
    score += s
    print(f"game {g+1}: {r} as {'white' if me else 'black'} in {board.fullmove_number} moves", flush=True)
sf.quit()
print(f"score {score}/{a.games} vs Stockfish skill {a.skill} (sims={a.sims}, adapter={a.adapter}, "
      f"value={'material' if a.material_value else (a.value_adapter or 'learned')})")
