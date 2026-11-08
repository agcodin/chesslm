"""Move-legality of engine.Policy, against a tiny fake model (random logits, no weights, no GPU).

Needs the real mlx.core for array ops, so these are skipped on Linux CI.
"""
import math

import chess
import pytest

from conftest import HAVE_MLX

pytestmark = pytest.mark.skipif(not HAVE_MLX, reason="MLX not installed (Linux CI)")

VOCAB = 256

CASES = {
    "start": chess.STARTING_FEN,
    "castling both ways": "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "promotions": "8/P6k/8/8/8/8/p6K/8 w - - 0 1",
    "promotion with capture": "1n5k/P7/8/8/8/8/8/K7 w - - 0 1",
    "en passant": "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "in check": "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",
}


@pytest.fixture(scope="module")
def policy():
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache  # noqa: F401  (ensures real module)

    import engine

    class Tok:
        def encode(self, text):
            return [ord(c) % VOCAB for c in text]

    class FakeModel:
        layers = [object(), object()]

        def __call__(self, x, cache=None):
            b, t = x.shape
            if cache is not None:
                for c in cache:
                    c.update_and_fetch(mx.zeros((b, 1, t, 2)), mx.zeros((b, 1, t, 2)))
            key = mx.random.key(int(x.sum().item()))   # logits depend on the tokens fed
            return mx.random.normal((b, t, VOCAB), key=key) * 2

    return engine.Policy((FakeModel(), Tok()), adapter=None)


@pytest.mark.parametrize("name", CASES)
def test_priors_cover_exactly_the_legal_moves(policy, name):
    board = chess.Board(CASES[name])
    legal = {m.uci() for m in board.legal_moves}
    p = policy.priors(board)
    assert set(p) == legal
    assert all(lp <= 0 and math.isfinite(lp) for lp in p.values())
    assert sum(math.exp(lp) for lp in p.values()) <= 1 + 1e-4   # a sub-distribution over full move strings


@pytest.mark.parametrize("name", CASES)
def test_trie_walk_reaches_the_same_legal_set(policy, name):
    board = chess.Board(CASES[name])
    assert set(policy.priors_trie(board)) == {m.uci() for m in board.legal_moves}


def test_special_moves_are_reachable(policy):
    castle = set(policy.priors(chess.Board(CASES["castling both ways"])))
    assert {"e1g1", "e1c1"} <= castle
    promos = set(policy.priors(chess.Board(CASES["promotions"])))
    assert {"a7a8q", "a7a8r", "a7a8b", "a7a8n"} <= promos
    assert "e5f6" in set(policy.priors(chess.Board(CASES["en passant"])))


def test_best_is_legal(policy):
    for fen in CASES.values():
        board = chess.Board(fen)
        assert policy.best(board) in board.legal_moves


def test_value_is_in_range(policy):
    assert -1.0 <= policy.value(chess.Board()) <= 1.0
