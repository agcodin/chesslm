#!/bin/bash
# Play ChessLM in your browser. Needs an Apple Silicon Mac (the model runs on MLX) and uv.
# First run downloads the base model (~3 GB) and a small embedding model for the coach.
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "ChessLM runs on MLX, which needs a Mac with Apple Silicon (M1 or later)." >&2
  exit 1
fi
if ! command -v uv > /dev/null; then
  echo "uv is not installed. Install it with:  brew install uv   (or see https://docs.astral.sh/uv/)" >&2
  exit 1
fi

uv sync --quiet

# Opening names for the coach (CC0, lichess-org/chess-openings). Optional: the bot plays without them.
if [[ ! -f kb/a.tsv ]]; then
  echo "Fetching the opening database..."
  for f in a b c d e; do
    curl -sfL "https://raw.githubusercontent.com/lichess-org/chess-openings/master/$f.tsv" -o "kb/$f.tsv" \
      || echo "  could not fetch $f.tsv; opening names will be missing" >&2
  done
fi

PORT=${PORT:-8000}
echo "Starting ChessLM on http://localhost:$PORT (the first start downloads the model, a few minutes)..."
( until curl -sf "http://localhost:$PORT/" > /dev/null; do sleep 2; done; open "http://localhost:$PORT" ) &
exec uv run uvicorn server:app --port "$PORT"
