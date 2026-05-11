"""backend/app.py — Flask: serve frontend + REST API"""
from flask import Flask, jsonify, send_from_directory
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
FRONTEND = ROOT / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND), static_url_path="")


# ── Pages ──────────────────────────────────────────────
@app.get("/")
def home():
    return send_from_directory(FRONTEND, "index.html")

@app.get("/<page>.html")
def page(page):
    return send_from_directory(FRONTEND, f"{page}.html")


# ── Static assets (plots) ──────────────────────────────
@app.get("/plots/<path:filename>")
def plots(filename):
    return send_from_directory(ROOT / "output_plots", filename)


# ── API ────────────────────────────────────────────────
def _csv(rel):
    return pd.read_csv(ROOT / rel).to_dict(orient="records")

@app.get("/api/cluster-profile")
def cluster_profile():
    return jsonify(_csv("output_plots/cluster_profile.csv"))

@app.get("/api/model-comparison")
def model_comparison():
    return jsonify(_csv("output_plots/supervised/model_comparison.csv"))

@app.get("/api/descriptive-stats")
def descriptive_stats():
    return jsonify(_csv("output_plots/descriptive_stats.csv"))

@app.get("/api/feature-importance")
def feature_importance():
    return jsonify(_csv("output_plots/supervised/feature_importance.csv"))

@app.get("/api/persona")
def persona():
    return jsonify(_csv("output_plots/persona_demo.csv"))

@app.get("/api/anomaly-summary")
def anomaly_summary():
    return jsonify(_csv("output_plots/anomaly_summary.csv"))


if __name__ == "__main__":
    app.run(port=8050, debug=True)
