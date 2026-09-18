import base64, json, pathlib
here = pathlib.Path(__file__).parent
repo = here.parent
games = json.load(open(here / "showcase_games.json"))   # from: uv run record_game.py --games 4
pieces = {f.stem: "data:image/svg+xml;base64," + base64.b64encode(f.read_bytes()).decode()
          for f in sorted((repo / "web/pieces").glob("*.svg"))}
s = (here / "template.html").read_text()
s = s.replace("__GAMES__", json.dumps(games, separators=(",", ":")))
s = s.replace("__PIECES__", json.dumps(pieces))
s = s.replace("REPO_URL", "https://github.com/agcodin/chesslm")
(here / "index.html").write_text(s)
print(len(s), "bytes;", len(games), "games;", len(pieces), "pieces")
