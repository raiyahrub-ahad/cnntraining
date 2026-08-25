@echo off
cd /d "%~dp0"

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Starting server at http://localhost:8000
echo.
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
