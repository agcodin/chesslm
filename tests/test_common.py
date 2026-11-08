import chess
import pytest

from common import BUCKETS, SCALE, bucket_letter, letter_values, prompt, value_prompt

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_21_letters_a_to_u():
    assert BUCKETS == 21
    letters = {bucket_letter(cp) for cp in range(-5000, 5001, 5)}
    assert letters == {chr(ord("a") + i) for i in range(21)}


def test_extremes_and_equal():
    assert bucket_letter(-10000) == "a"
    assert bucket_letter(10000) == "u"
    assert bucket_letter(0) == "k"


def test_monotonic():
    cps = range(-3000, 3001, 7)
    idx = [ord(bucket_letter(cp)) for cp in cps]
    assert idx == sorted(idx)


def test_letter_values_endpoints_and_symmetry():
    vals = letter_values()
    assert len(vals) == BUCKETS
    assert vals[0] == -1.0 and vals[-1] == 1.0 and vals[10] == 0.0
    assert vals == pytest.approx([-v for v in reversed(vals)])


@pytest.mark.parametrize("cp", [-2000, -400, -75, 0, 40, 400, 900, 1500])
def test_round_trip_within_one_bucket(cp):
    """Decoding a letter returns the squashed value of the bucket it came from, to within half a bucket."""
    import math
    decoded = letter_values()[ord(bucket_letter(cp)) - ord("a")]
    assert abs(decoded - math.tanh(cp / SCALE)) <= 1 / (BUCKETS - 1) + 1e-9


def test_mate_score_is_top_bucket():
    assert bucket_letter(10000) == "u" and bucket_letter(-10000) == "a"


def test_prompt_formats():
    b = chess.Board()
    assert prompt(b) == f"FEN: {START_FEN}\nBest move:"
    assert value_prompt(b) == f"FEN: {START_FEN}\nEval:"
