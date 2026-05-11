"""app.py — Entry point for local run and Vercel deployment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backend.db import init_db
from backend.app import app

init_db()

if __name__ == "__main__":
    print("🐱 Silent Salesman → http://127.0.0.1:8050")
    app.run(port=8050, debug=True)
