"""FastAPI endpoints with the model and coach mocked out: no weights, no embeddings, no GPU."""
import importlib
import sys

import chess
import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

import coach  # noqa: E402
import engine  # noqa: E402
from conftest import ROOT  # noqa: E402


class FakePolicy:
    def __init__(self, *a, **k):
        pass

    def priors(self, board):
        return {m.uci(): -1.0 for m in board.legal_moves}

    def value(self, board):
        return 0.0


class FakeCoach:
    def __init__(self, *a, **k):
        pass

    def explain(self, board, move, search_info=""):
        return {"explanation": "stub", "facts": ["stub fact"], "opening": None, "retrieved": []}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.chdir(ROOT)   # server mounts web/pieces relative to the repo root
    monkeypatch.setattr(engine, "Policy", FakePolicy)
    monkeypatch.setattr(coach, "Coach", FakeCoach)
    monkeypatch.delitem(sys.modules, "server", raising=False)
    server = importlib.import_module("server")
    yield TestClient(server.app)
    sys.modules.pop("server", None)


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200 and "<html" in r.text.lower()


def test_move_returns_a_legal_reply(client):
    r = client.post("/api/move", json={"moves": ["e2e4"], "sims": 4})
    assert r.status_code == 200
    body = r.json()
    board = chess.Board()
    board.push_uci("e2e4")
    assert chess.Move.from_uci(body["uci"]) in board.legal_moves
    board.push_uci(body["uci"])
    assert body["fen"] == board.fen()
    assert body["explanation"] == "stub" and body["game_over"] is None
    assert 1 <= len(body["candidates"]) <= 5


def test_illegal_move_is_400(client):
    assert client.post("/api/move", json={"moves": ["e2e5"]}).status_code == 400


def test_unparseable_move_is_400(client):
    assert client.post("/api/move", json={"moves": ["nonsense"]}).status_code == 400


def test_game_over_short_circuits(client):
    fools_mate = ["f2f3", "e7e5", "g2g4", "d8h4"]
    body = client.post("/api/move", json={"moves": fools_mate}).json()
    assert body["game_over"] == "0-1" and "uci" not in body


def test_sims_are_clamped(client):
    # sims=0 is clamped up to 1 rather than erroring.
    assert client.post("/api/move", json={"moves": [], "sims": 0}).status_code == 200
