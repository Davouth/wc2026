"""
Statball — API Server
Expone el modelo ML como endpoint REST para el dashboard.

Arrancar con:
    python3 api.py

Luego el dashboard en http://localhost:5500 (o file://) llama a:
    http://localhost:8000/predict?home_id=64&away_id=65&league=PL&season=2024
"""
from fastapi import FastAPI, Query, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pickle
import math
import sqlite3
import pandas as pd
import uvicorn
import os
import sys
import urllib.request
import urllib.parse
import urllib.error
import json as _json

try:
    import anthropic as _anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    import random
    HAS_NUMPY = False

# Asegurarse de que el directorio del script esté en el path
sys.path.insert(0, os.path.dirname(__file__))
from features import get_team_form, get_h2h
from database import get_conn

MODEL_PATH = os.path.join(os.path.dirname(__file__), "statball_model.pkl")

app = FastAPI(title="Statball ML API", version="1.0")

# Permitir llamadas desde el dashboard (file:// y cualquier origen)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Cargar modelo al arrancar
try:
    with open(MODEL_PATH, "rb") as f:
        BUNDLE = pickle.load(f)
    print(f"✅ Modelo cargado — entrenado con {BUNDLE['n_train']} partidos (accuracy {BUNDLE['accuracy']:.1%})")
except FileNotFoundError:
    BUNDLE = None
    print("⚠️  Modelo no encontrado — ejecutá python3 model.py --train primero")

FEATURE_COLS = [
    "home_form_pts", "away_form_pts",
    "home_gpg", "away_gpg",
    "home_gcpg", "away_gcpg",
    "home_points", "away_points",
    "h2h_home_wins", "h2h_draws", "h2h_away_wins",
    "home_ha_gpg", "away_ha_gpg",
    "home_ha_gcpg", "away_ha_gcpg",
    "pts_diff", "form_diff", "gpg_diff", "gcpg_diff",
    "h2h_dominance",
]

def engineer(row: dict) -> dict:
    row["pts_diff"]      = row["home_points"] - row["away_points"]
    row["form_diff"]     = row["home_form_pts"] - row["away_form_pts"]
    row["gpg_diff"]      = row["home_gpg"] - row["away_gpg"]
    row["gcpg_diff"]     = row["home_gcpg"] - row["away_gcpg"]
    h2h_total            = row["h2h_home_wins"] + row["h2h_draws"] + row["h2h_away_wins"]
    row["h2h_dominance"] = (row["h2h_home_wins"] - row["h2h_away_wins"]) / (h2h_total + 1)
    return row

def poisson_over(lam: float, k: int) -> float:
    return 1 - sum(math.exp(-lam) * lam**i / math.factorial(i) for i in range(k + 1))

def top_score(xg_h: float, xg_a: float):
    best = {"score": "1-0", "prob": 0.0}
    for h in range(6):
        for a in range(6):
            p = (math.exp(-xg_h) * xg_h**h / math.factorial(h)) * \
                (math.exp(-xg_a) * xg_a**a / math.factorial(a))
            if p > best["prob"]:
                best = {"score": f"{h}-{a}", "prob": round(p * 100, 1)}
    return best

@app.get("/predict")
def predict(
    home_id: int = Query(..., description="ID del equipo local"),
    away_id: int = Query(..., description="ID del equipo visitante"),
    league:  str = Query("PL",   description="Código de liga"),
    season:  str = Query("2024", description="Temporada"),
):
    if BUNDLE is None:
        return {"error": "Modelo no disponible — ejecutá python3 model.py --train"}

    conn = get_conn()
    c = conn.cursor()

    # Última fecha disponible como referencia
    c.execute("SELECT MAX(utc_date) FROM matches WHERE status='FINISHED'")
    last_date = c.fetchone()[0] or "2099-01-01"

    # Features
    hf      = get_team_form(home_id, last_date, league, season, conn)
    af      = get_team_form(away_id, last_date, league, season, conn)
    hf_home = get_team_form(home_id, last_date, league, season, conn, home_away="home")
    af_away = get_team_form(away_id, last_date, league, season, conn, home_away="away")
    h2h     = get_h2h(home_id, away_id, last_date, conn)

    c.execute("""SELECT SUM(CASE
        WHEN home_id=? AND home_goals>away_goals THEN 3
        WHEN home_id=? AND home_goals=away_goals THEN 1
        WHEN away_id=? AND away_goals>home_goals THEN 3
        WHEN away_id=? AND away_goals=home_goals THEN 1
        ELSE 0 END) FROM matches WHERE (home_id=? OR away_id=?) AND season=?""",
        (home_id,)*4 + (home_id, home_id, season))
    h_pts = c.fetchone()[0] or 0

    c.execute("""SELECT SUM(CASE
        WHEN home_id=? AND home_goals>away_goals THEN 3
        WHEN home_id=? AND home_goals=away_goals THEN 1
        WHEN away_id=? AND away_goals>home_goals THEN 3
        WHEN away_id=? AND away_goals=home_goals THEN 1
        ELSE 0 END) FROM matches WHERE (home_id=? OR away_id=?) AND season=?""",
        (away_id,)*4 + (away_id, away_id, season))
    a_pts = c.fetchone()[0] or 0

    # Nombres de equipos
    c.execute("SELECT name, short_name FROM teams WHERE id=?", (home_id,))
    h_row = c.fetchone() or ("Equipo Local", "Local")
    c.execute("SELECT name, short_name FROM teams WHERE id=?", (away_id,))
    a_row = c.fetchone() or ("Equipo Visitante", "Visit.")

    conn.close()

    row = {
        "home_form_pts": hf["pts"],    "away_form_pts": af["pts"],
        "home_gpg":  hf["gpg"],        "away_gpg":  af["gpg"],
        "home_gcpg": hf["gcpg"],       "away_gcpg": af["gcpg"],
        "home_points": h_pts,           "away_points": a_pts,
        "h2h_home_wins": h2h["h_wins"],"h2h_draws": h2h["draws"],
        "h2h_away_wins": h2h["a_wins"],
        "home_ha_gpg":  hf_home["gpg"],"away_ha_gpg":  af_away["gpg"],
        "home_ha_gcpg": hf_home["gcpg"],"away_ha_gcpg": af_away["gcpg"],
    }
    row = engineer(row)
    df = pd.DataFrame([row])
    X = df[FEATURE_COLS].fillna(0)

    clf, reg, btts_clf, le = BUNDLE["clf"], BUNDLE["reg"], BUNDLE["btts"], BUNDLE["le"]

    probs    = clf.predict_proba(X)[0]
    classes  = le.classes_
    prob_map = dict(zip(classes, probs))

    xg_total = float(reg.predict(X)[0])
    xg_h     = round(xg_total * 0.55, 2)
    xg_a     = round(xg_total * 0.45, 2)
    btts     = float(btts_clf.predict_proba(X)[0][1])

    best_score = top_score(xg_h, xg_a)

    return {
        "home_id":    home_id,
        "away_id":    away_id,
        "home_name":  h_row[1] or h_row[0],
        "away_name":  a_row[1] or a_row[0],
        "home_pct":   round(prob_map.get("H", 0) * 100, 1),
        "draw_pct":   round(prob_map.get("D", 0) * 100, 1),
        "away_pct":   round(prob_map.get("A", 0) * 100, 1),
        "btts_pct":   round(btts * 100, 1),
        "over15_pct": round(poisson_over(xg_total, 1) * 100, 1),
        "over25_pct": round(poisson_over(xg_total, 2) * 100, 1),
        "over35_pct": round(poisson_over(xg_total, 3) * 100, 1),
        "xg_home":    xg_h,
        "xg_away":    xg_a,
        "xg_total":   round(xg_total, 2),
        "top_score":  best_score["score"],
        "top_score_pct": best_score["prob"],
        "model_accuracy": round(BUNDLE["accuracy"] * 100, 1),
        "data_points": BUNDLE["n_train"],
        "source": "xgboost_model"
    }

@app.get("/teams")
def teams(league: str = Query("PL")):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id, name, short_name FROM teams WHERE league=? ORDER BY name", (league,))
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "short": r[2]} for r in rows]

@app.get("/team-stats/{team_id}")
def team_stats(team_id: int, season: str = Query("2025")):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT name, short_name FROM teams WHERE id=?", (team_id,))
    t = c.fetchone() or ("Unknown", "?")
    c.execute("""
        SELECT home_id, away_id, home_goals, away_goals FROM matches
        WHERE (home_id=? OR away_id=?) AND status='FINISHED' AND season=?
        ORDER BY utc_date DESC LIMIT 20
    """, (team_id, team_id, season))
    rows = c.fetchall()
    conn.close()
    if not rows:
        return {"team_id": team_id, "name": t[0], "short_name": t[1] or t[0][:3].upper(),
                "matches": 0, "avg_gpg": 0, "avg_gcpg": 0, "cs_pct": 0, "btts_pct": 0, "form": [],
                "home_w": 0, "home_d": 0, "home_l": 0, "away_w": 0, "away_d": 0, "away_l": 0}
    form, gs, gcl = [], [], []
    hw = hd = hl = aw = ad = al = cs = btts = 0
    for hid, aid, hg, ag in rows:
        ih = (hid == team_id)
        gf = hg if ih else ag
        gc = ag if ih else hg
        gs.append(gf); gcl.append(gc)
        if gc == 0: cs += 1
        if gf > 0 and gc > 0: btts += 1
        if gf > gc:
            form.append("W")
            if ih: hw += 1
            else: aw += 1
        elif gf == gc:
            form.append("D")
            if ih: hd += 1
            else: ad += 1
        else:
            form.append("L")
            if ih: hl += 1
            else: al += 1
    n = len(rows)
    return {
        "team_id": team_id, "name": t[0], "short_name": t[1] or t[0][:3].upper(),
        "matches": n, "avg_gpg": round(sum(gs)/n, 2), "avg_gcpg": round(sum(gcl)/n, 2),
        "cs_pct": round(cs/n*100, 1), "btts_pct": round(btts/n*100, 1), "form": form[:10],
        "home_w": hw, "home_d": hd, "home_l": hl, "away_w": aw, "away_d": ad, "away_l": al
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": BUNDLE is not None, "accuracy": round(BUNDLE["accuracy"]*100,1) if BUNDLE else None, "ai": HAS_ANTHROPIC}

class ChatRequest(BaseModel):
    question: str
    context: str
    api_key: str = ""
    system: str = ""  # custom system prompt from frontend (includes memory instructions)

@app.post("/chat")
def chat(req: ChatRequest):
    if not HAS_ANTHROPIC:
        return {"error": "anthropic no instalado — ejecutá: pip3 install anthropic"}
    key = req.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return {"error": "Necesitás una API key de Anthropic. Creá una gratis en console.anthropic.com y pegala en el campo 'IA:' del dashboard."}
    try:
        client = _anthropic.Anthropic(api_key=key)
        system_prompt = req.system if req.system else (
            "Sos un analista de fútbol experto. Respondés en español rioplatense, "
            "de forma directa y concisa. Tomás posición clara basándote en los datos."
        )
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=700,
            system=system_prompt,
            messages=[{"role": "user", "content": f"Datos del partido:\n{req.context}\n\nPregunta: {req.question}"}]
        )
        return {"text": msg.content[0].text}
    except _anthropic.AuthenticationError:
        return {"error": "API key inválida. Verificá en console.anthropic.com → API Keys."}
    except Exception as e:
        return {"error": str(e)}

@app.get("/simulate")
def simulate(
    home_id: int = Query(..., description="ID del equipo local"),
    away_id: int = Query(..., description="ID del equipo visitante"),
    league:  str = Query("PL",   description="Código de liga"),
    season:  str = Query("2024", description="Temporada"),
    n_sims:  int = Query(10000,  description="Número de simulaciones"),
):
    N_SIMS = max(1000, min(n_sims, 50000))
    HOME_ADVANTAGE = 1.12

    conn = get_conn()
    c = conn.cursor()

    # ── League averages ──────────────────────────────────────────────────────
    c.execute("""
        SELECT AVG(home_goals + away_goals)
        FROM matches
        WHERE league=? AND season=? AND status='FINISHED'
          AND home_goals IS NOT NULL
    """, (league, season))
    row = c.fetchone()
    league_avg_gpg = (row[0] or 2.6) / 2.0  # per-team per-game average

    confidence_pts = 0  # track data availability

    # ── Team stats helper ────────────────────────────────────────────────────
    def team_season_stats(team_id):
        """Return (gpg, gcpg, n_matches) for team in league/season."""
        c.execute("""
            SELECT
                SUM(CASE WHEN home_id=? THEN home_goals ELSE away_goals END) AS gf,
                SUM(CASE WHEN home_id=? THEN away_goals ELSE home_goals END) AS ga,
                COUNT(*) AS n
            FROM matches
            WHERE (home_id=? OR away_id=?)
              AND league=? AND season=? AND status='FINISHED'
              AND home_goals IS NOT NULL
        """, (team_id, team_id, team_id, team_id, league, season))
        r = c.fetchone()
        n = r[2] or 0
        gpg  = (r[0] or 0) / n if n else league_avg_gpg
        gcpg = (r[1] or 0) / n if n else league_avg_gpg
        return gpg, gcpg, n

    h_gpg,  h_gcpg,  h_n  = team_season_stats(home_id)
    a_gpg,  a_gcpg,  a_n  = team_season_stats(away_id)

    # Confidence based on data volume (full marks at 15+ matches each)
    confidence_pts += min(h_n, 15) / 15 * 40   # up to 40 pts
    confidence_pts += min(a_n, 15) / 15 * 40   # up to 40 pts

    # ── Recent form (last 5 games) ───────────────────────────────────────────
    c.execute("SELECT MAX(utc_date) FROM matches WHERE status='FINISHED'")
    last_date = c.fetchone()[0] or "2099-01-01"

    hf = get_team_form(home_id, last_date, league, season, conn)
    af = get_team_form(away_id, last_date, league, season, conn)

    def form_weight(form_pts):
        # form_pts is 0-15 (5 games × 3 pts max)
        pts = min(max(form_pts or 0, 0), 15)
        return 0.85 + 0.15 * (pts / 15)

    fw_h = form_weight(hf.get("pts", 7.5))
    fw_a = form_weight(af.get("pts", 7.5))

    # Confidence: form data available → extra 10 pts each
    if hf.get("n", 0) >= 3:
        confidence_pts += 5
    if af.get("n", 0) >= 3:
        confidence_pts += 5

    # Remaining 10 pts: league avg reliable when ≥20 finished matches
    c.execute("""
        SELECT COUNT(*) FROM matches
        WHERE league=? AND season=? AND status='FINISHED' AND home_goals IS NOT NULL
    """, (league, season))
    league_n = c.fetchone()[0] or 0
    confidence_pts += min(league_n, 20) / 20 * 10

    # ── Fatigue: days since last match ──────────────────────────────────────
    def days_rest(team_id):
        c.execute("""
            SELECT MAX(utc_date) FROM matches
            WHERE (home_id=? OR away_id=?) AND status='FINISHED'
        """, (team_id, team_id))
        r = c.fetchone()[0]
        if not r:
            return -1
        try:
            from datetime import date
            last = date.fromisoformat(r[:10])
            today = date.fromisoformat(last_date[:10])
            return (today - last).days
        except Exception:
            return -1

    home_days_rest = days_rest(home_id)
    away_days_rest = days_rest(away_id)

    def fatigue_factor(days):
        """Slight penalty for very short rest (<3 days), neutral otherwise."""
        if days < 0:
            return 1.0
        if days <= 2:
            return 0.94
        if days <= 4:
            return 0.97
        return 1.0

    conn.close()

    # ── Dixon-Coles xG calculation ───────────────────────────────────────────
    safe_avg = league_avg_gpg if league_avg_gpg > 0 else 1.3

    attack_h  = h_gpg  / safe_avg
    defense_a = a_gcpg / safe_avg
    attack_a  = a_gpg  / safe_avg
    defense_h = h_gcpg / safe_avg

    xg_home = attack_h * defense_a * safe_avg * HOME_ADVANTAGE * fw_h * fatigue_factor(home_days_rest)
    xg_away = attack_a * defense_h * safe_avg * (1 / HOME_ADVANTAGE) * fw_a * fatigue_factor(away_days_rest)

    # Guard against extreme values
    xg_home = max(0.1, min(xg_home, 6.0))
    xg_away = max(0.1, min(xg_away, 6.0))

    # ── Monte Carlo simulation ───────────────────────────────────────────────
    if HAS_NUMPY:
        goals_h = np.random.poisson(xg_home, N_SIMS)
        goals_a = np.random.poisson(xg_away, N_SIMS)
    else:
        import random as _r
        import math as _m
        def _poisson_sample(lam):
            L = _m.exp(-lam)
            k, p = 0, 1.0
            while p > L:
                k += 1
                p *= _r.random()
            return k - 1
        goals_h = [_poisson_sample(xg_home) for _ in range(N_SIMS)]
        goals_a = [_poisson_sample(xg_away) for _ in range(N_SIMS)]
        goals_h = goals_h  # keep as list; indexing works fine below

    # Outcome counters
    if HAS_NUMPY:
        home_wins  = int(np.sum(goals_h > goals_a))
        draws      = int(np.sum(goals_h == goals_a))
        away_wins  = int(np.sum(goals_h < goals_a))
        btts       = int(np.sum((goals_h > 0) & (goals_a > 0)))
        total      = goals_h + goals_a
        over15     = int(np.sum(total > 1))
        over25     = int(np.sum(total > 2))
        over35     = int(np.sum(total > 3))

        # Score distribution
        from collections import Counter
        scores = Counter(zip(goals_h.tolist(), goals_a.tolist()))

        # Goal distributions (0..5, 6+)
        def goal_dist(arr):
            dist = []
            for k in range(6):
                dist.append(round(float(np.sum(arr == k)) / N_SIMS * 100, 2))
            dist.append(round(float(np.sum(arr >= 6)) / N_SIMS * 100, 2))
            return dist

        gd_h = goal_dist(goals_h)
        gd_a = goal_dist(goals_a)
    else:
        from collections import Counter
        home_wins = sum(1 for h, a in zip(goals_h, goals_a) if h > a)
        draws     = sum(1 for h, a in zip(goals_h, goals_a) if h == a)
        away_wins = sum(1 for h, a in zip(goals_h, goals_a) if h < a)
        btts      = sum(1 for h, a in zip(goals_h, goals_a) if h > 0 and a > 0)
        totals    = [h + a for h, a in zip(goals_h, goals_a)]
        over15    = sum(1 for t in totals if t > 1)
        over25    = sum(1 for t in totals if t > 2)
        over35    = sum(1 for t in totals if t > 3)
        scores    = Counter(zip(goals_h, goals_a))

        def goal_dist(arr):
            dist = [round(arr.count(k) / N_SIMS * 100, 2) for k in range(6)]
            dist.append(round(sum(1 for x in arr if x >= 6) / N_SIMS * 100, 2))
            return dist

        gd_h = goal_dist(goals_h)
        gd_a = goal_dist(goals_a)

    # Build score distribution (top 15)
    score_dist = []
    for (gh, ga), cnt in scores.most_common(15):
        pct = round(cnt / N_SIMS * 100, 2)
        if gh > ga:
            outcome = "H"
        elif gh == ga:
            outcome = "D"
        else:
            outcome = "A"
        score_dist.append({"score": f"{gh}-{ga}", "pct": pct, "outcome": outcome})

    return {
        "xg_home":        round(xg_home, 3),
        "xg_away":        round(xg_away, 3),
        "home_pct":       round(home_wins / N_SIMS * 100, 1),
        "draw_pct":       round(draws     / N_SIMS * 100, 1),
        "away_pct":       round(away_wins / N_SIMS * 100, 1),
        "btts_pct":       round(btts      / N_SIMS * 100, 1),
        "over15_pct":     round(over15    / N_SIMS * 100, 1),
        "over25_pct":     round(over25    / N_SIMS * 100, 1),
        "over35_pct":     round(over35    / N_SIMS * 100, 1),
        "score_distribution": score_dist,
        "goal_dist_home": gd_h,
        "goal_dist_away": gd_a,
        "confidence":     round(min(confidence_pts, 100), 1),
        "n_sims":         N_SIMS,
        "home_days_rest": home_days_rest,
        "away_days_rest": away_days_rest,
    }


@app.get("/fd/{path:path}")
def football_data_proxy(path: str, request: Request):
    """Proxy para football-data.org — evita CORS en el browser (sin deps extra)."""
    api_key = request.headers.get("X-Auth-Token", "")
    params = dict(request.query_params)
    qs = urllib.parse.urlencode(params)
    url = f"https://api.football-data.org/v4/{path}"
    if qs:
        url += "?" + qs
    req = urllib.request.Request(url, headers={"X-Auth-Token": api_key})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return JSONResponse(content=_json.loads(resp.read()))
    except urllib.error.HTTPError as e:
        return JSONResponse(content=_json.loads(e.read()), status_code=e.code)


if __name__ == "__main__":
    print("\n🚀 Statball ML API arrancando...")
    print("   Dashboard: abrí index.html en el browser")
    print("   API docs:  http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
