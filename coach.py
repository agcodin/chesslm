"""RAG coach: retrieve openings (exact position match) and chess concepts (vector search),
then have the base instruct model explain a move grounded in that context."""
import csv, glob, io, json
import chess, chess.pgn
import numpy as np
from fastembed import TextEmbedding
from mlx_lm import load, generate

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def load_openings():
    book = {}
    for path in sorted(glob.glob("kb/[a-e].tsv")):
        for row in csv.DictReader(open(path), delimiter="\t"):
            game = chess.pgn.read_game(io.StringIO(row["pgn"]))
            board = game.end().board()
            book[board.epd()] = f'{row["eco"]} {row["name"]}'  # later, longer lines overwrite: most specific name
    return book


def features(board: chess.Board, move: chess.Move) -> list[str]:
    """Plain-language facts about a move, used both as the retrieval query and as LLM context."""
    f, side = [], board.turn
    piece = board.piece_at(move.from_square)
    name = chess.piece_name(piece.piece_type)
    if board.is_capture(move):
        cap = board.piece_at(move.to_square)
        f.append(f"captures a {chess.piece_name(cap.piece_type) if cap else 'pawn en passant'}")
    if board.is_castling(move):
        f.append("castles for king safety")
    if move.promotion:
        f.append("promotes a pawn")
    after = board.copy()
    after.push(move)
    if after.is_checkmate():
        f.append("delivers checkmate")
    elif after.is_check():
        f.append("gives check")
    targets = [after.piece_at(s) for s in after.attacks(move.to_square)]
    valuable = [t for t in targets if t and t.color != side and t.piece_type != chess.PAWN]
    if len(valuable) >= 2:
        f.append(f"the {name} attacks two pieces at once (fork)")
    elif valuable and not after.is_check():
        t = valuable[0]
        sq = next(s for s in after.attacks(move.to_square) if after.piece_at(s) == t)
        f.append(f"the {name} attacks the enemy {chess.piece_name(t.piece_type)} on {chess.square_name(sq)}")
    for sq in chess.SQUARES:
        p = after.piece_at(sq)
        if p and p.color != side and after.is_pinned(not side, sq) and p.piece_type != chess.KING:
            f.append(f"the enemy {chess.piece_name(p.piece_type)} on {chess.square_name(sq)} is pinned")
    hanging = [chess.square_name(s) for s in chess.SQUARES
               if (p := after.piece_at(s)) and p.color == side and p.piece_type != chess.KING
               and after.is_attacked_by(not side, s) and not after.is_attacked_by(side, s)]
    if hanging:
        f.append(f"leaves pieces undefended on {', '.join(hanging)}")
    if piece.piece_type == chess.PAWN and chess.square_file(move.to_square) in (3, 4) and board.fullmove_number <= 10:
        f.append("fights for the center with a pawn")
    if piece.piece_type in (chess.KNIGHT, chess.BISHOP) and chess.square_rank(move.from_square) in (0, 7):
        f.append(f"develops the {name}")
    mat = sum(v * (len(after.pieces(t, side)) - len(after.pieces(t, not side))) for t, v in VALUES.items())
    f.append(f"material balance for the mover is {mat:+d}")
    return f


class Coach:
    def __init__(self, model="Qwen/Qwen2.5-1.5B-Instruct"):
        self.book = load_openings()
        self.docs = [json.loads(l) for l in open("kb/concepts.jsonl")]
        self.embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
        self.doc_vecs = np.array(list(self.embedder.embed([d["title"] + ": " + d["text"] for d in self.docs])))
        self.model, self.tok = load(model)

    def retrieve(self, query: str, k=3):
        q = np.array(list(self.embedder.query_embed(query)))[0]
        scores = self.doc_vecs @ q
        return [self.docs[i] | {"score": float(scores[i])} for i in np.argsort(-scores)[:k]]

    def opening(self, board):
        # Walk back to the last position that is in the book.
        b = board.copy()
        while True:
            if b.epd() in self.book:
                return self.book[b.epd()]
            if not b.move_stack:
                return None
            b.pop()

    def explain(self, board: chess.Board, move: chess.Move, search_info: str = ""):
        facts = features(board, move)
        after = board.copy()
        after.push(move)
        opening = self.opening(after)
        docs = self.retrieve(f"{board.san(move)}: " + "; ".join(facts), k=2)
        context = "\n".join(f"- {d['title']}: {d['text']}" for d in docs)
        msg = (f"You are a friendly chess coach. Explain in 2-3 sentences why {board.san(move)} "
               f"was played. State only what the move facts say. Mention a principle only if a move fact "
               f"directly shows it; never claim threats, pins, forks or skewers that are not in the facts.\n"
               f"Position (FEN): {board.fen()}\nMove facts: {'; '.join(facts)}\n"
               + (f"Opening: {opening}\n" if opening else "")
               + (f"Engine search: {search_info}\n" if search_info else "")
               + f"Relevant chess principles:\n{context}")
        text = generate(self.model, self.tok,
                        self.tok.apply_chat_template([{"role": "user", "content": msg}],
                                                     add_generation_prompt=True, tokenize=False),
                        max_tokens=140)
        return {"explanation": text.strip(), "facts": facts, "opening": opening,
                "retrieved": [{"title": d["title"], "score": round(d["score"], 3)} for d in docs]}
