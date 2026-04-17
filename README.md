```
           ██████╗██████╗ ██╗ ██████╗██╗  ██╗   ██╗████████╗██╗ ██████╗███████╗
          ██╔════╝██╔══██╗██║██╔════╝██║  ╚██╗ ██╔╝╚══██╔══╝██║██╔════╝██╔════╝
          ██║     ██████╔╝██║██║     ██║   ╚████╔╝    ██║   ██║██║     ███████╗
          ██║     ██╔══██╗██║██║     ██║    ╚██╔╝     ██║   ██║██║     ╚════██║
          ╚██████╗██║  ██║██║╚██████╗███████╗██║      ██║   ██║╚██████╗███████║
           ╚═════╝╚═╝  ╚═╝╚═╝ ╚═════╝╚══════╝╚═╝      ╚═╝   ╚═╝ ╚═════╝╚══════╝
                                                                        
```                                                                                                                                    
                                                                                                                                    
                                                                                                                                    
---

## Overview

Criclytics is a full-stack web application that estimates the win probability of a batting team at any point during an IPL match. Enter the current match situation — score, overs bowled, wickets fallen, and (for the second innings) the target — and the model instantly returns how likely the batting team is to win.

It supports **both innings**:
- **1st Innings** — predicts win likelihood based on projected total given current scoring rate and resources remaining
- **2nd Innings (Chase)** — uses required run rate vs. current run rate, wickets in hand, and balls left to estimate chase probability

---

## Features

- **Dual-innings ML models** — separate pipelines trained per innings with appropriate feature sets, useful from ball one rather than just mid-chase
- **Real match data** — 2nd innings model trained on 260,920 ball-by-ball IPL deliveries aggregated into per-over snapshots
- **Strong evaluation** — 81.5% accuracy and 0.90 ROC-AUC on a held-out test set, with well-calibrated probabilities
- **Team & venue awareness** — all 10 current IPL franchises and their home grounds encoded as features
- **Input validation** — both client-side (React) and server-side (Flask) guards against invalid match states

---

## ML Approach

### Data Pipeline (2nd Innings)

```
Raw CSV (260,920 rows)
  → Filter to inning == 2
  → Group by (match_id, over)
  → Compute cumulative runs, wickets, balls
  → Derive CRR, RRR, run_rate_diff, balls_remaining
  → Binary label: did batting team win?
  → Train/test split (80/20, stratified)
```

### Model Architecture

Both innings use a **Logistic Regression** wrapped in a scikit-learn Pipeline:

```
ColumnTransformer
  ├── OneHotEncoder   →  batting_team, bowling_team, venue
  └── passthrough     →  numerical features

LogisticRegression (C=1.0, max_iter=1000)
```

Logistic Regression is well-suited here because match win probability should be a smooth, monotonic function of features like run rate differential — not a jagged decision tree boundary. It also produces well-calibrated probabilities out of the box, which the calibration curve from the evaluation notebook confirms.

### Engineered Features

| Feature | Description |
|---------|-------------|
| `current_run_rate` | Runs per over scored so far |
| `required_run_rate` | Runs per over needed from here (2nd innings) |
| `run_rate_diff` | Gap between RRR and CRR — strongest predictor |
| `balls_remaining` | Resource pressure indicator |
| `wickets_left` | In-hand resources |
| `batting_team`, `bowling_team`, `venue` | Team & venue identity (one-hot encoded) |

### 1st Innings Model

Trained on synthetically generated match situations using a domain-heuristic win function based on projected final score vs. the IPL average (~165 runs). Features: `current_score`, `balls_left`, `wickets_left`, `current_run_rate`.

---

## Project Structure

```
criclytics/
├── backend/
│   ├── app.py                  # Flask API — /predict, /teams, /venues, /health
│   ├── train_model.py          # Training script for both innings models
│   ├── Model_evaluation.ipynb  # EDA, training, and full evaluation notebook
│   ├── IPL_Data.csv            # Ball-by-ball IPL dataset
│   ├── model.pkl               # Trained 2nd innings pipeline
│   ├── model_innings1.pkl      # Trained 1st innings pipeline
│   └── encoder.pkl             # Reference encoder
│
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── App.jsx             # Main UI — form, validation, result display
        ├── main.jsx            # React entry point
        └── styles.css          # Neobrutalism design system
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+

### Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Although the pre-trained `.pkl` files are included. To retrain from scratch:

```bash
python train_model.py
```

The flask API will be available at `http://localhost:5000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| ML / Data | scikit-learn, pandas, numpy |
| Backend | Flask, Flask-CORS |
| Frontend | React 19, Vite |
| Styling | Custom CSS (Neobrutalism) |

---

*Built with Flask + React + scikit-learn*
