"""Training-data formatting. Prompt/completion shapes are checked without Stockfish; the one
test that labels a position with the real engine is skipped when Stockfish is not installed."""
import json
import shutil

import chess
import pytest

from common import bucket_letter, prompt, value_prompt

FENS = [
    chess.STARTING_FEN,
    "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
]


@pytest.mark.parametrize("fen", FENS)
def test_rows_are_valid_jsonl_with_expected_shape(fen):
    board = chess.Board(fen)
    move = next(iter(board.legal_moves))
    rows = [
        {"prompt": prompt(board), "completion": " " + move.uci()},
        {"prompt": value_prompt(board), "completion": " " + bucket_letter(35)},
    ]
    for r in rows:
        r2 = json.loads(json.dumps(r))
        assert r2 == r
        assert r["prompt"].startswith("FEN: " + fen) and r["completion"].startswith(" ")
    assert rows[0]["prompt"].endswith("Best move:") and rows[1]["prompt"].endswith("Eval:")
    assert chess.Move.from_uci(rows[0]["completion"].strip()) in board.legal_moves
    assert rows[1]["completion"].strip() in "abcdefghijklmnopqrstu"


@pytest.mark.skipif(shutil.which("stockfish") is None, reason="Stockfish not installed")
def test_stockfish_labels_mate_in_one():
    import chess.engine
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    with chess.engine.SimpleEngine.popen_uci("stockfish") as eng:
        info = eng.analyse(board, chess.engine.Limit(depth=8))
    assert info["pv"][0].uci() == "a1a8"
    assert bucket_letter(info["score"].relative.score(mate_score=10000)) == "u"
