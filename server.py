"""Web UI backend: play against ChessLM, see search visits and the RAG coach's explanation."""
import os
import time
import chess
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from engine import Policy, search
from coach import Coach

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SHIPPED_ADAPTER = "weights/chesslm-expert"


def load_policy():
    """CHESSLM_MODEL wins; then a locally merged models/best; otherwise the base model plus the adapter
    shipped in the repo, so a fresh clone plays the trained bot without retraining anything."""
    if os.environ.get("CHESSLM_MODEL"):
        return Policy(os.environ["CHESSLM_MODEL"], adapter=None)
    if os.path.isdir("models/best"):
        return Policy("models/best", adapter=None)
    return Policy(BASE_MODEL, adapter=SHIPPED_ADAPTER)


app = FastAPI()
app.mount("/pieces", StaticFiles(directory="web/pieces"), name="pieces")
policy = load_policy()
coach = Coach(BASE_MODEL)


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
