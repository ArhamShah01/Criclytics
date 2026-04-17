"""
Flask backend for IPL Win Predictor.
Supports both 1st innings (batting) and 2nd innings (chase) predictions.
Loads pre-trained models and exposes a /predict endpoint.
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Load models ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
MODEL_INN1_PATH = os.path.join(BASE_DIR, "model_innings1.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoder.pkl")

with open(MODEL_PATH, "rb") as f:
    model_innings2 = pickle.load(f)

model_innings1 = None
if os.path.exists(MODEL_INN1_PATH):
    with open(MODEL_INN1_PATH, "rb") as f:
        model_innings1 = pickle.load(f)

encoder = None
if os.path.exists(ENCODER_PATH):
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)


# ── Teams & venues ──────────────────────────────────────────────────
TEAMS = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
]

VENUES = [
    "Wankhede Stadium, Mumbai",
    "M. A. Chidambaram Stadium, Chennai",
    "Eden Gardens, Kolkata",
    "Arun Jaitley Stadium, Delhi",
    "Rajiv Gandhi Intl. Cricket Stadium, Hyderabad",
    "M. Chinnaswamy Stadium, Bengaluru",
    "Narendra Modi Stadium, Ahmedabad",
    "Sawai Mansingh Stadium, Jaipur",
    "Punjab Cricket Association Stadium, Mohali",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
]


# ── Feature engineering ─────────────────────────────────────────────
def compute_features_innings2(input_json: dict) -> pd.DataFrame:
    """
    2nd innings (chase): compute features from target + current state.

    Returns DataFrame with columns:
        batting_team, bowling_team, venue,
        runs_left, balls_left, wickets_left,
        current_run_rate, required_run_rate
    """
    batting_team = input_json["batting_team"]
    bowling_team = input_json["bowling_team"]
    venue = input_json["venue"]
    current_score = float(input_json["current_score"])
    overs = float(input_json["overs"])
    wickets = int(input_json["wickets"])
    target = float(input_json["target"])

    balls_bowled = int(overs) * 6 + int(round((overs % 1) * 10))
    balls_left = max(120 - balls_bowled, 0)
    runs_left = max(target - current_score, 0)
    wickets_left = max(10 - wickets, 0)

    current_run_rate = round(current_score / overs, 2) if overs > 0 else 0.0
    required_run_rate = (
        round(runs_left / (balls_left / 6), 2) if balls_left > 0 else 99.0
    )

    return pd.DataFrame([{
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "venue": venue,
        "runs_left": runs_left,
        "balls_left": balls_left,
        "wickets_left": wickets_left,
        "current_run_rate": current_run_rate,
        "required_run_rate": required_run_rate,
    }])


def compute_features_innings1(input_json: dict) -> pd.DataFrame:
    """
    1st innings (batting first): compute features from current state.
    No target available.

    Returns DataFrame with columns:
        batting_team, bowling_team, venue,
        current_score, balls_left, wickets_left,
        current_run_rate
    """
    batting_team = input_json["batting_team"]
    bowling_team = input_json["bowling_team"]
    venue = input_json["venue"]
    current_score = float(input_json["current_score"])
    overs = float(input_json["overs"])
    wickets = int(input_json["wickets"])

    balls_bowled = int(overs) * 6 + int(round((overs % 1) * 10))
    balls_left = max(120 - balls_bowled, 0)
    wickets_left = max(10 - wickets, 0)

    current_run_rate = round(current_score / overs, 2) if overs > 0 else 0.0

    return pd.DataFrame([{
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "venue": venue,
        "current_score": current_score,
        "balls_left": balls_left,
        "wickets_left": wickets_left,
        "current_run_rate": current_run_rate,
    }])


# ── Endpoints ────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """Return batting team's win probability for either innings."""
    data = request.get_json(force=True)

    innings = int(data.get("innings", 2))

    # Common required fields
    required = ["batting_team", "bowling_team", "current_score",
                 "overs", "wickets", "venue"]

    # 2nd innings also needs target
    if innings == 2:
        required.append("target")

    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        if innings == 1:
            if model_innings1 is None:
                return jsonify({"error": "1st innings model not loaded"}), 500
            features = compute_features_innings1(data)
            proba = model_innings1.predict_proba(features)[0]
        else:
            features = compute_features_innings2(data)
            proba = model_innings2.predict_proba(features)[0]

        win_prob = float(proba[1])
        return jsonify({"win_probability": round(win_prob, 4)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/teams", methods=["GET"])
def get_teams():
    """Return list of valid teams."""
    return jsonify({"teams": TEAMS})


@app.route("/venues", methods=["GET"])
def get_venues():
    """Return list of valid venues."""
    return jsonify({"venues": VENUES})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
