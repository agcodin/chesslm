"""Measure a model on four axes at once, so compression can be judged on more than perplexity.

The axes are deliberately ordered from surface competence to actual skill:

  ppl        token-level perplexity on held-out move completions -- the metric compression papers
             usually report, and the one that averages away the thing we care about
  legal      greedy decode with no constraints: is the emitted text a legal UCI move? surface
             competence, the model still "sounds like" a chess engine
  top1       agreement with Stockfish depth 12, legality enforced by the trie -- real move skill
  value      correlation and sign agreement of the learned evaluation against Stockfish

The claim worth testing is that these do not decay together.
"""
import argparse, json, re
import chess
import mlx.core as mx
import numpy as np
from common import letter_values, prompt as move_prompt, value_prompt
from engine import Policy


def perplexity(policy, rows) -> float:
    """Mean token NLL over the completion only, so the shared FEN prompt cannot flatter the score."""
    total, count = 0.0, 0
    for r in rows:
        p_ids = policy.tok.encode(r["prompt"])
        c_ids = policy.tok.encode(r["completion"])
        ids = p_ids + c_ids
        logits = policy.model(mx.array([ids[:-1]]))[0]
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tgt = mx.array(ids[1:])
        nll = -mx.take_along_axis(logp, tgt[:, None], axis=-1)[:, 0]
        total += float(nll[len(p_ids) - 1:].sum())
        count += len(c_ids)
    return float(np.exp(total / max(count, 1)))


UCI_RE = re.compile(r"[a-h][1-8][a-h][1-8][qrbn]?")


def legal_rate(policy, boards, max_tokens=8) -> float:
    """Unconstrained greedy decode: does the model emit a legal move on its own?

    This is the axis the trie normally hides. It measures whether the model still behaves like a
    chess engine at the surface, independent of whether the move it picks is any good.
    """
    from mlx_lm.models.cache import make_prompt_cache
    eos = policy.tok.eos_token_id
    ok = 0
    for board in boards:
        cache = make_prompt_cache(policy.model)
        cur, out = policy.tok.encode(move_prompt(board)), []
        for _ in range(max_tokens):
            logits = policy.model(mx.array([cur]), cache=cache)[0, -1]
            nxt = int(mx.argmax(logits))
            if nxt == eos:
                break
            out.append(nxt)
            cur = [nxt]
        m = UCI_RE.match(policy.tok.decode(out).strip())
        if m:
            try:
                ok += chess.Move.from_uci(m.group()) in board.legal_moves
            except ValueError:
                pass
    return ok / max(len(boards), 1)


def fen_of(row) -> str:
    return row["prompt"].split("\n")[0][5:]


def evaluate_all(model, adapter, n_move, n_value, skip_legal=False) -> dict:
    policy = Policy(model, adapter=adapter)

    move_rows = [json.loads(l) for l in open("data/test.jsonl")][:n_move]
    boards = [chess.Board(fen_of(r)) for r in move_rows]

    out = {"n_move": len(move_rows)}
    out["ppl"] = round(perplexity(policy, move_rows), 4)
    if not skip_legal:
        out["legal"] = round(legal_rate(policy, boards), 4)
    hit = sum(policy.best(b).uci() == r["completion"].strip() for b, r in zip(boards, move_rows))
    out["top1"] = round(hit / len(move_rows), 4)

    if n_value:
        vals = letter_values()
        vrows = [json.loads(l) for l in open("data_v3/test.jsonl")]
        vrows = [r for r in vrows if "\nEval:" in r["prompt"]][:n_value]
        pred = np.array([policy.value(chess.Board(fen_of(r))) for r in vrows])
        true = np.array([vals[ord(r["completion"].strip()) - ord("a")] for r in vrows])
        out.update(n_value=len(vrows),
                   value_corr=round(float(np.corrcoef(pred, true)[0, 1]), 3),
                   value_mae=round(float(np.abs(pred - true).mean()), 3),
                   value_sign=round(float(((pred > 0) == (true > 0)).mean()), 3))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="base model id, or a directory written by compress.py")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n-move", type=int, default=300)
    ap.add_argument("--n-value", type=int, default=200)
    ap.add_argument("--skip-legal", action="store_true")
    ap.add_argument("--tag", default=None, help="label carried through into the JSON line")
    ap.add_argument("--out", default=None, help="append the result as one JSON line to this file")
    a = ap.parse_args()

    res = evaluate_all(a.model, a.adapter, a.n_move, a.n_value, a.skip_legal)
    res = {"tag": a.tag or a.model, **res}
    line = json.dumps(res)
    print(line)
    if a.out:
        with open(a.out, "a") as f:
            f.write(line + "\n")
