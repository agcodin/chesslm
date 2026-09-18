"""Estimate ChessLM's Elo on a ladder anchored to Stockfish's calibrated UCI_Elo 1320.

Stockfish's strength limiter bottoms out at 1320 and ChessLM loses every game there, so weaker
anchors (random mover, 1-ply Stockfish, Stockfish skill 0) are placed on the same scale by playing
each other, and every game is fitted jointly with a Bradley-Terry / Elo model.

Also reports blunder rate: for each ChessLM move, how many centipawns it lost versus Stockfish's
best move (depth 10). A move losing >= 300cp counts as a blunder. Unlike win/loss this separates
versions of the model even when every game is lost.

Usage: uv run rating.py --games 10 --sims 16 [--material-value] [--out runs/rating_x.json]
"""
import argparse, json, math, random, time
import chess, chess.engine
from engine import Policy, search

ANCHOR, ANCHOR_ELO = "sf_1320", 1320.0
BLUNDER_CP = 300


def sf_player(engine, **opts):
    def play(board):
        return engine.play(board, chess.engine.Limit(**opts)).move
    return play


def play_game(white, black, max_plies=240):
    board = chess.Board()
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        board.push((white if board.turn == chess.WHITE else black)(board))
    r = board.result(claim_draw=True)
    return (1.0 if r == "1-0" else 0.0 if r == "0-1" else 0.5), board


def fit_elo(games, anchor=ANCHOR, anchor_elo=ANCHOR_ELO, iters=4000, lr=4.0):
    """games: list of (a, b, score_for_a). Maximum likelihood with one virtual draw per pair so
    all-loss records give a finite (conservative) estimate instead of minus infinity."""
    players = sorted({p for g in games for p in g[:2]})
    pairs = {tuple(sorted(g[:2])) for g in games}
    data = list(games) + [(a, b, 0.5) for a, b in pairs]
    r = {p: anchor_elo for p in players}
    for _ in range(iters):
        grad = {p: 0.0 for p in players}
        for a, b, s in data:
            e = 1 / (1 + 10 ** ((r[b] - r[a]) / 400))
            grad[a] += s - e
            grad[b] -= s - e
        for p in players:
            if p != anchor:
                r[p] += lr * grad[p]
    return r


def bootstrap(games, player, n=200, seed=0):
    """90% interval by resampling games within each matchup, so every resample keeps every pairing."""
    rng = random.Random(seed)
    by_pair = {}
    for g in games:
        by_pair.setdefault(tuple(sorted(g[:2])), []).append(g)
    est = []
    for _ in range(n):
        sample = [rng.choice(gs) for gs in by_pair.values() for _ in gs]
        est.append(fit_elo(sample, iters=1500)[player])
    est.sort()
    return est[int(0.05 * n)], est[int(0.95 * n)]


def blunder_stats(boards, color_of, analyst):
    losses = []
    for board, color in zip(boards, color_of):
        replay = chess.Board()
        for mv in board.move_stack:
            if replay.turn == color and not replay.is_game_over():
                lim = chess.engine.Limit(depth=10)
                best = analyst.analyse(replay, lim)["score"].pov(color).score(mate_score=3000)
                replay.push(mv)
                after = analyst.analyse(replay, lim)["score"].pov(color).score(mate_score=3000)
                losses.append(min(max(best - after, 0), 1000))
            else:
                replay.push(mv)
    if not losses:
        return {}
    return {"moves": len(losses), "acpl": round(sum(losses) / len(losses), 1),
            "blunder_rate": round(sum(l >= BLUNDER_CP for l in losses) / len(losses), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/best")
    ap.add_argument("--games", type=int, default=10, help="ChessLM games per opponent")
    ap.add_argument("--anchor-games", type=int, default=20)
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--material-value", action="store_true")
    ap.add_argument("--value-mode", choices=["learned", "material", "blend"], default=None)
    ap.add_argument("--anchors-file", default="runs/anchor_games.json", help="reuse anchor games across runs")
    ap.add_argument("--out", default="runs/rating.json")
    a = ap.parse_args()

    engines = [chess.engine.SimpleEngine.popen_uci("stockfish") for _ in range(4)]
    sf1320, sf_skill0, sf_ply1, analyst = engines
    sf1320.configure({"UCI_LimitStrength": True, "UCI_Elo": 1320})
    sf_skill0.configure({"Skill Level": 0})
    rng = random.Random(0)
    anchors = {
        "random": lambda b: rng.choice(list(b.legal_moves)),
        "sf_depth1": sf_player(sf_ply1, depth=1),
        "sf_skill0": sf_player(sf_skill0, time=0.05),
        ANCHOR: sf_player(sf1320, time=0.1),
    }
    order = ["random", "sf_depth1", "sf_skill0", ANCHOR]

    # Anchor ladder: adjacent rungs, colours alternated. Cached so every model is rated on the same ladder.
    try:
        anchor_games = [tuple(g) for g in json.load(open(a.anchors_file))]
        print(f"reusing {len(anchor_games)} anchor games from {a.anchors_file}")
    except (FileNotFoundError, json.JSONDecodeError):
        anchor_games = []
        for lo, hi in zip(order, order[1:]):
            for g in range(a.anchor_games):
                w, b = (lo, hi) if g % 2 == 0 else (hi, lo)
                s, _ = play_game(anchors[w], anchors[b])
                anchor_games.append((w, b, s))
            won = sum(s if w == lo else 1 - s for w, b, s in anchor_games if lo in (w, b) and hi in (w, b))
            print(f"{lo} vs {hi}: {won}/{a.anchor_games}", flush=True)
        json.dump(anchor_games, open(a.anchors_file, "w"))

    policy = Policy(a.model, None)
    name = "chesslm"
    mode = a.value_mode or ("material" if a.material_value else "learned")
    me = lambda b: search(policy, b, sims=a.sims, value_mode=mode)[0]
    games, boards, colors = [], [], []
    t0 = time.time()
    for opp in ["random", "sf_depth1", "sf_skill0"]:
        pts = 0.0
        for g in range(a.games):
            if g % 2 == 0:
                s, board = play_game(me, anchors[opp]); games.append((name, opp, s)); colors.append(chess.WHITE); pts += s
            else:
                s, board = play_game(anchors[opp], me); games.append((opp, name, s)); colors.append(chess.BLACK); pts += 1 - s
            boards.append(board)
        print(f"chesslm vs {opp}: {pts}/{a.games}  ({time.time() - t0:.0f}s)", flush=True)

    all_games = anchor_games + games
    r = fit_elo(all_games)
    lo, hi = bootstrap(all_games, name)
    blunders = blunder_stats(boards, colors, analyst)
    for e in engines:
        e.quit()
    result = {"model": a.model, "sims": a.sims, "value": mode,
              "elo": round(r[name]), "elo_90ci": [round(lo), round(hi)],
              "anchors": {p: round(r[p]) for p in order}, **blunders,
              "results": {opp: sum((s if w == name else 1 - s) for w, b, s in games if opp in (w, b))
                          for opp in ["random", "sf_depth1", "sf_skill0"]}}
    json.dump(result, open(a.out, "w"), indent=1)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
