import chess

BUCKETS = 21          # value buckets, encoded as single letters a..u  (a = lost, k = equal, u = won)
SCALE = 400.0         # centipawns at which tanh saturates

def prompt(board: chess.Board) -> str:
    # Legality is enforced by engine.Policy's trie over legal moves, so the prompt stays short.
    return f"FEN: {board.fen()}\nBest move:"

def value_prompt(board: chess.Board) -> str:
    return f"FEN: {board.fen()}\nEval:"

def bucket_letter(cp: float) -> str:
    """Centipawns (side to move) -> one of 21 letters, squashed so big advantages compress."""
    import math
    v = math.tanh(cp / SCALE)
    return chr(ord("a") + round((v + 1) * (BUCKETS - 1) / 2))

def letter_values() -> list[float]:
    """The value in [-1, 1] each bucket letter stands for."""
    return [(i * 2 / (BUCKETS - 1)) - 1 for i in range(BUCKETS)]
