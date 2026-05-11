"""backend/db.py — Initialize SQLite DB and populate from CSV files."""
import sqlite3
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent.parent
DB_PATH = Path("/tmp/analytics.db")

CSV_MAP = {
    "cluster_results":   BASE / "output_plots" / "cluster_profile.csv",
    "model_results":     BASE / "output_plots" / "supervised" / "model_comparison.csv",
    "feature_importance": BASE / "output_plots" / "supervised" / "feature_importance.csv",
    "anomaly_summary":   BASE / "output_plots" / "anomaly_summary.csv",
}


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    for table, csv_path in CSV_MAP.items():
        df = pd.read_csv(csv_path)
        df.to_sql(table, conn, if_exists="replace", index=False)
    conn.close()
    print(f"DB initialized at {DB_PATH}")


def query(table: str) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(f"SELECT * FROM {table}").fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
