import { useState } from "react";
import "./styles.css";

const TEAMS = [
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
];

const VENUES = [
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
];

const TEAM_COLORS = {
  "Chennai Super Kings": "#FFC107",
  "Delhi Capitals": "#004BA0",
  "Gujarat Titans": "#1C2C5B",
  "Kolkata Knight Riders": "#3A225D",
  "Lucknow Super Giants": "#00B2FF",
  "Mumbai Indians": "#004BA0",
  "Punjab Kings": "#ED1B24",
  "Rajasthan Royals": "#EA1A85",
  "Royal Challengers Bengaluru": "#D4171E",
  "Sunrisers Hyderabad": "#FF822A",
};

const API_URL = "http://localhost:5000";

function App() {
  const [innings, setInnings] = useState(2);
  const [form, setForm] = useState({
    batting_team: "",
    bowling_team: "",
    current_score: "",
    overs: "",
    wickets: "",
    target: "",
    venue: "",
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
    setError("");
  };

  const switchInnings = (inn) => {
    setInnings(inn);
    setResult(null);
    setError("");
  };

  const validate = () => {
    if (!form.batting_team) return "Select batting team";
    if (!form.bowling_team) return "Select bowling team";
    if (form.batting_team === form.bowling_team) return "Teams must be different";
    if (!form.venue) return "Select venue";
    if (form.current_score === "" || isNaN(form.current_score) || Number(form.current_score) < 0)
      return "Enter valid score";
    if (form.overs === "" || isNaN(form.overs) || Number(form.overs) < 0 || Number(form.overs) > 20)
      return "Overs must be 0–20";
    if (form.wickets === "" || isNaN(form.wickets) || Number(form.wickets) < 0 || Number(form.wickets) > 10)
      return "Wickets must be 0–10";
    if (innings === 2) {
      if (form.target === "" || isNaN(form.target) || Number(form.target) < 1)
        return "Enter valid target";
    }
    if (Number(form.overs) === 0 && Number(form.current_score) > 0)
      return "Score must be 0 at 0 overs";
    return null;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const err = validate();
    if (err) {
      setError(err);
      return;
    }

    setLoading(true);
    setResult(null);
    setError("");

    const payload = {
      innings,
      batting_team: form.batting_team,
      bowling_team: form.bowling_team,
      current_score: Number(form.current_score),
      overs: Number(form.overs),
      wickets: Number(form.wickets),
      venue: form.venue,
    };
    if (innings === 2) {
      payload.target = Number(form.target);
    }

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (data.error) {
        setError(data.error);
      } else {
        setResult({
          win: data.win_probability,
          lose: 1 - data.win_probability,
          battingTeam: form.batting_team,
          bowlingTeam: form.bowling_team,
        });
      }
    } catch {
      setError("Could not reach server. Is Flask running on port 5000?");
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setForm({
      batting_team: "",
      bowling_team: "",
      current_score: "",
      overs: "",
      wickets: "",
      target: "",
      venue: "",
    });
    setResult(null);
    setError("");
  };

  return (
    <div className="app">
      {/* Decorative elements */}
      <div className="deco deco-1"></div>
      <div className="deco deco-2"></div>
      <div className="deco deco-3"></div>

      <header className="header">
        <div className="header-badge">🏏</div>
        <h1 className="title">IPL WIN PREDICTOR</h1>
        <p className="subtitle">PREDICT THE OUTCOME OF ANY MATCH SITUATION</p>
      </header>

      <main className="main-container">
        <form className="form-card" onSubmit={handleSubmit}>

          {/* Innings Toggle */}
          <div className="innings-toggle">
            <button
              type="button"
              className={`innings-btn ${innings === 1 ? "innings-active" : ""}`}
              onClick={() => switchInnings(1)}
            >
              1ST INNINGS
            </button>
            <button
              type="button"
              className={`innings-btn ${innings === 2 ? "innings-active" : ""}`}
              onClick={() => switchInnings(2)}
            >
              2ND INNINGS
            </button>
          </div>

          <h2 className="form-heading">
            {innings === 1 ? "🏏 BATTING FIRST" : "⚡ CHASING TARGET"}
          </h2>

          {/* Teams row */}
          <div className="row">
            <div className="field">
              <label htmlFor="batting_team">BATTING TEAM</label>
              <select
                id="batting_team"
                name="batting_team"
                value={form.batting_team}
                onChange={handleChange}
              >
                <option value="">— Select —</option>
                {TEAMS.map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </div>
            <div className="vs-badge">VS</div>
            <div className="field">
              <label htmlFor="bowling_team">BOWLING TEAM</label>
              <select
                id="bowling_team"
                name="bowling_team"
                value={form.bowling_team}
                onChange={handleChange}
              >
                <option value="">— Select —</option>
                {TEAMS.map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Venue */}
          <div className="field full-width">
            <label htmlFor="venue">VENUE</label>
            <select
              id="venue"
              name="venue"
              value={form.venue}
              onChange={handleChange}
            >
              <option value="">— Select Stadium —</option>
              {VENUES.map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </div>

          {/* Match numbers */}
          <div className={`stats-grid ${innings === 1 ? "stats-grid-3" : ""}`}>
            {innings === 2 && (
              <div className="field">
                <label htmlFor="target">TARGET</label>
                <input
                  id="target"
                  name="target"
                  type="number"
                  min="1"
                  placeholder="185"
                  value={form.target}
                  onChange={handleChange}
                />
              </div>
            )}
            <div className="field">
              <label htmlFor="current_score">SCORE</label>
              <input
                id="current_score"
                name="current_score"
                type="number"
                min="0"
                placeholder="120"
                value={form.current_score}
                onChange={handleChange}
              />
            </div>
            <div className="field">
              <label htmlFor="overs">OVERS</label>
              <input
                id="overs"
                name="overs"
                type="number"
                min="0"
                max="20"
                step="0.1"
                placeholder="12.3"
                value={form.overs}
                onChange={handleChange}
              />
            </div>
            <div className="field">
              <label htmlFor="wickets">WICKETS</label>
              <input
                id="wickets"
                name="wickets"
                type="number"
                min="0"
                max="10"
                placeholder="3"
                value={form.wickets}
                onChange={handleChange}
              />
            </div>
          </div>

          {innings === 1 && (
            <div className="innings-hint">
              ⓘ 1st innings — no target needed. Prediction based on projected score.
            </div>
          )}

          {error && <div className="error-box">{error}</div>}

          <div className="button-row">
            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? (
                <span className="spinner-wrap">
                  <span className="spinner"></span> PREDICTING…
                </span>
              ) : (
                "🔮 PREDICT WIN"
              )}
            </button>
            <button type="button" className="btn btn-secondary" onClick={resetForm}>
              ↺ RESET
            </button>
          </div>
        </form>

        {/* Results */}
        {result && (
          <div className="result-card">
            <h2 className="result-heading">📊 PREDICTION RESULT</h2>

            <div className="prob-section">
              <div className="team-prob">
                <span
                  className="team-dot"
                  style={{ background: TEAM_COLORS[result.battingTeam] || "#FFD600" }}
                ></span>
                <span className="team-name">{result.battingTeam}</span>
                <span className="team-pct">{(result.win * 100).toFixed(1)}%</span>
              </div>
              <div className="bar-track">
                <div
                  className="bar-fill bar-fill-win"
                  style={{
                    width: `${result.win * 100}%`,
                    background: TEAM_COLORS[result.battingTeam] || "#FFD600",
                  }}
                ></div>
                <div
                  className="bar-fill bar-fill-lose"
                  style={{
                    width: `${result.lose * 100}%`,
                    background: TEAM_COLORS[result.bowlingTeam] || "#1E1E1E",
                  }}
                ></div>
              </div>
              <div className="team-prob">
                <span
                  className="team-dot"
                  style={{ background: TEAM_COLORS[result.bowlingTeam] || "#1E1E1E" }}
                ></span>
                <span className="team-name">{result.bowlingTeam}</span>
                <span className="team-pct">{(result.lose * 100).toFixed(1)}%</span>
              </div>
            </div>

            <div className="big-prob">
              <div className="big-prob-label">
                {result.win >= 0.5 ? result.battingTeam : result.bowlingTeam}
              </div>
              <div className="big-prob-number">
                {(Math.max(result.win, result.lose) * 100).toFixed(1)}%
              </div>
              <div className="big-prob-sub">CHANCE OF WINNING</div>
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>BUILT WITH FLASK + REACT + SCIKIT-LEARN &nbsp;|&nbsp; IPL WIN PREDICTOR 2026</p>
      </footer>
    </div>
  );
}

export default App;
