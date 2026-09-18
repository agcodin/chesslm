"""Web UI backend: play against ChessLM, see search visits and the RAG coach's explanation."""
import time
import chess
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from engine import Policy, search
from coach import Coach

app = FastAPI()
app.mount("/pieces", StaticFiles(directory="web/pieces"), name="pieces")
policy = Policy(adapter="adapters_best")
coach = Coach()


class MoveReq(BaseModel):
    moves: list[str]  # full game so far in UCI, so the server stays stateless
    sims: int = 32


@app.get("/")
def index():
    return FileResponse("web/index.html")


@app.post("/api/move")
async def move(req: MoveReq):  # MLX streams are thread-bound; stay on the loop thread
    board = chess.Board()
    try:
        for m in req.moves:
            board.push_uci(m)
    except (ValueError, AssertionError):
        raise HTTPException(400, "illegal move")
    if board.is_game_over(claim_draw=True):
        return {"game_over": board.result(claim_draw=True), "fen": board.fen()}
    t = time.time()
    best, visits = search(policy, board, sims=max(1, min(req.sims, 128)))
    top = sorted(visits.items(), key=lambda kv: -kv[1])[:5]
    candidates = [{"san": board.san(m), "uci": m.uci(), "visits": n} for m, n in top]
    info = ", ".join(f"{c['san']} {c['visits']} visits" for c in candidates)
    coaching = coach.explain(board, best, info)
    san = board.san(best)
    board.push(best)
    return {"uci": best.uci(), "san": san, "fen": board.fen(), "think_ms": int((time.time() - t) * 1000),
            "candidates": candidates,
            "game_over": board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else None,
            **coaching}
