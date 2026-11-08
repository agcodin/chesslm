import chess

import coach


def facts(fen, uci):
    board = chess.Board(fen)
    return coach.features(board, chess.Move.from_uci(uci))


def test_checkmate_and_material():
    f = facts("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8")
    assert "delivers checkmate" in f
    assert "material balance for the mover is +2" in f


def test_capture_and_castling():
    assert "captures a pawn" in facts("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5")
    assert "castles for king safety" in facts("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1")


def test_en_passant_and_promotion():
    assert "captures a pawn en passant" in facts("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6")
    assert "promotes a pawn" in facts("7k/P7/8/8/8/8/8/K7 w - - 0 1", "a7a8q")


def test_development_and_center():
    assert "develops the knight" in facts(chess.STARTING_FEN, "g1f3")
    assert "fights for the center with a pawn" in facts(chess.STARTING_FEN, "e2e4")


def test_fork_detected():
    # Knight to c7 forks king e8 and rook a8.
    f = facts("r3k3/8/8/3N4/8/8/8/4K3 w - - 0 1", "d5c7")
    assert any("fork" in x or "gives check" in x for x in f)


def test_opening_book_prefers_longest_line(tmp_path, monkeypatch):
    (tmp_path / "kb").mkdir()
    (tmp_path / "kb" / "a.tsv").write_text(
        "eco\tname\tpgn\n"
        "B00\tKing's Pawn\t1. e4\n"
        "C20\tKing's Pawn Game: Open\t1. e4 e5\n"
    )
    monkeypatch.chdir(tmp_path)
    book = coach.load_openings()
    assert len(book) == 2

    class Stub(coach.Coach):
        def __init__(self):
            self.book = book

    c = Stub()
    board = chess.Board()
    for m in ["e2e4", "e7e5", "g1f3"]:   # past the book: falls back to the last known position
        board.push_uci(m)
    assert c.opening(board) == "C20 King's Pawn Game: Open"
    assert c.opening(chess.Board()) is None
