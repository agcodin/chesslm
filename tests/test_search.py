import math
import random

import chess
import pytest

import engine
from engine import evaluate, search

BACK_RANK_MATE = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"   # Ra8# is the only mate
FENS = [
    chess.STARTING_FEN,
    BACK_RANK_MATE,
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 b kq - 0 7",   # both sides can castle later
    "8/P6k/8/8/8/8/p6K/8 w - - 0 1",                                          # promotions available
    "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",                                       # castling both ways
]


class StubPolicy:
    """Quacks like engine.Policy for search(): random but deterministic priors over legal moves."""

    def __init__(self, seed=0, boost=None):
        self.rng = random.Random(seed)
        self.calls = 0
        self.boost = boost   # {uci: prior} overrides, to model a policy that ranks a move highly

    def priors(self, board):
        self.calls += 1
        out = {m.uci(): math.log(self.rng.uniform(0.01, 0.2)) for m in board.legal_moves}
        for uci, p in (self.boost or {}).items():
            if uci in out:
                out[uci] = math.log(p)
        return out

    def value(self, board):
        return 0.0


def test_finds_forced_mate_in_one():
    """With the mating move given a modest prior (0.3 vs <=0.2 for the rest), the terminal -1 for the
    mated side makes it the most-visited move."""
    board = chess.Board(BACK_RANK_MATE)
    best, visits = search(StubPolicy(boost={"a1a8": 0.3}), board, sims=64, top_k=64, learned_value=False)
    assert best == chess.Move.from_uci("a1a8")
    assert visits[best] == max(visits.values())


@pytest.mark.parametrize("fen", FENS)
@pytest.mark.parametrize("mode", ["material", "learned", "blend"])
def test_never_returns_illegal_move(fen, mode):
    board = chess.Board(fen)
    best, visits = search(StubPolicy(seed=1), board, sims=24, value_mode=mode)
    assert best in board.legal_moves
    assert all(m in board.legal_moves for m in visits)


def test_tiny_prior_can_hide_a_mate_at_low_sims():
    """Documents a real limit: unvisited children score 0 (no first-play urgency), so a mate whose
    prior is tiny can go unvisited within a small budget."""
    board = chess.Board(BACK_RANK_MATE)
    _, visits = search(StubPolicy(boost={"a1a8": 0.0001}), board, sims=32, top_k=64, learned_value=False)
    assert visits[chess.Move.from_uci("a1a8")] == 0


def test_respects_sims_budget():
    board = chess.Board()
    policy = StubPolicy()
    _, visits = search(policy, board, sims=20, learned_value=False)
    # The first simulation only expands the root, so child visits total sims - 1.
    assert sum(visits.values()) == 19
    assert policy.calls <= 20


def test_top_k_limits_root_children():
    _, visits = search(StubPolicy(), chess.Board(), sims=30, top_k=5, learned_value=False)
    assert len(visits) == 5


def test_search_does_not_mutate_input_board():
    board = chess.Board(FENS[2])
    fen_before, stack_before = board.fen(), list(board.move_stack)
    search(StubPolicy(), board, sims=16, learned_value=False)
    assert board.fen() == fen_before and list(board.move_stack) == stack_before


def test_search_is_deterministic_for_fixed_priors():
    a = search(StubPolicy(seed=3), chess.Board(FENS[2]), sims=32, learned_value=False)
    b = search(StubPolicy(seed=3), chess.Board(FENS[2]), sims=32, learned_value=False)
    assert a == b


def test_evaluate_terminal_and_material():
    mated = chess.Board("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1")
    assert evaluate(mated) == -1.0
    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert evaluate(stalemate) == 0.0
    up_a_queen = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    assert evaluate(up_a_queen) > 0.5
    assert evaluate(chess.Board("4k3/8/8/8/8/8/8/3QK3 b - - 0 1")) < -0.5


def test_evaluate_uses_policy_value_when_given():
    class P:
        def value(self, board):
            return 0.25
    assert evaluate(chess.Board(), P()) == 0.25
    assert hasattr(engine, "Policy")
