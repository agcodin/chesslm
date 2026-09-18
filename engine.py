"""LLM policy + PUCT tree search.

The fine-tuned LLM supplies move priors and, through a second prompt, a learned evaluation
used at the leaves. A handcrafted material + mobility evaluation remains as a baseline.
"""
import math
import chess
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
from common import BUCKETS, letter_values, prompt, value_prompt

PRUNE_LOGP = math.log(1e-4)  # below this, stop expanding a prefix; its moves share the floor
VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3.2, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


class Policy:
    def __init__(self, model="Qwen/Qwen2.5-1.5B-Instruct", adapter="adapters", value_adapter=None):
        """value_adapter loads a second model used only for evaluation, so one adapter can stay
        specialised on moves and another on value instead of trading one off against the other."""
        self.model, self.tok = load(model, adapter_path=adapter)
        self.value_model = load(model, adapter_path=value_adapter)[0] if value_adapter else self.model
        self.bucket_ids = mx.array([self.tok.encode(" " + chr(ord("a") + i))[0] for i in range(BUCKETS)])
        self.bucket_vals = mx.array(letter_values())

    def value(self, board: chess.Board) -> float:
        """Learned evaluation in [-1, 1] for the side to move: mean of the bucket distribution."""
        logits = self.value_model(mx.array([self.tok.encode(value_prompt(board))]))[0, -1]
        p = mx.softmax(logits[self.bucket_ids])
        return float((p * self.bucket_vals).sum())

    def _step(self, ids, cache):
        logits = self.model(mx.array([ids]), cache=cache)[0, -1]
        return logits - mx.logsumexp(logits)

    def priors(self, board: chess.Board) -> dict:
        """log p(move | position) for every legal move, via a trie walk over move characters."""
        moves = [m.uci() for m in board.legal_moves]
        cache = make_prompt_cache(self.model)
        logp = self._step(self.tok.encode(prompt(board)), cache)
        out = {}

        def walk(prefix, logp, acc):
            group = [m for m in moves if m.startswith(prefix)]
            if len(group) == 1 and group[0] == prefix:
                out[prefix] = acc
                return
            text = lambda c: (" " + c) if not prefix else c
            chars = sorted({m[len(prefix)] for m in group if len(m) > len(prefix)})
            ids = {c: self.tok.encode(text(c)) for c in chars}
            mx.eval(logp)
            for c in chars:
                a = acc + logp[ids[c][0]].item()
                if a < PRUNE_LOGP:
                    for m in group:
                        if m.startswith(prefix + c):
                            out[m] = a
                    continue
                nxt = self._step(ids[c], cache)
                walk(prefix + c, nxt, a)
                trim_prompt_cache(cache, len(ids[c]))
            if prefix in moves:  # e.g. never happens for UCI, kept for safety
                out[prefix] = acc

        walk("", logp, 0.0)
        return out

    def best(self, board):
        p = self.priors(board)
        return chess.Move.from_uci(max(p, key=p.get))


def evaluate(board: chess.Board, policy: "Policy | None" = None) -> float:
    """Value in [-1, 1] from the side to move's perspective.

    With a policy, use its learned evaluation; otherwise fall back to material + mobility.
    """
    if board.is_checkmate():
        return -1.0
    if board.is_game_over(claim_draw=True):
        return 0.0
    if policy is not None:
        return policy.value(board)
    mat = sum(v * (len(board.pieces(p, board.turn)) - len(board.pieces(p, not board.turn)))
              for p, v in VALUES.items())
    mob = board.legal_moves.count()
    board.push(chess.Move.null())
    mob -= board.legal_moves.count()
    board.pop()
    return math.tanh((mat + 0.05 * mob) / 6)


class Node:
    __slots__ = ("prior", "children", "n", "w")

    def __init__(self, prior):
        self.prior, self.children, self.n, self.w = prior, None, 0, 0.0


def search(policy: Policy, board: chess.Board, sims=48, c_puct=1.5, top_k=8, learned_value=True):
    """PUCT search; returns (best move, {move: visits}). Expansions keep only the top_k priors."""
    root = Node(1.0)

    def expand(node, b):
        lp = policy.priors(b)
        top = sorted(lp.items(), key=lambda kv: -kv[1])[:top_k]
        z = sum(math.exp(v) for _, v in top)
        node.children = {chess.Move.from_uci(m): Node(math.exp(v) / z) for m, v in top}

    for _ in range(sims):
        b, node, path = board.copy(stack=False), root, [root]
        while node.children:
            sq = math.sqrt(node.n + 1)
            mv, node = max(node.children.items(),
                           key=lambda kv: (-kv[1].w / kv[1].n if kv[1].n else 0.0)
                           + c_puct * kv[1].prior * sq / (1 + kv[1].n))
            b.push(mv)
            path.append(node)
        v = evaluate(b, policy if learned_value else None)
        if not b.is_game_over(claim_draw=True):
            expand(node, b)
        # v is from the perspective of the side to move at the leaf; alternate sign going up.
        for nd in reversed(path):
            nd.n += 1
            nd.w += v
            v = -v
    visits = {m: c.n for m, c in root.children.items()}
    return max(visits, key=visits.get), visits
