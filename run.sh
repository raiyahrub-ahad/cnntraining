#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Starting server at http://localhost:8000"
echo ""
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
