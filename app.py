import numpy as np
import os
from flask import Flask, request, jsonify
from xgboost import XGBClassifier

app = Flask(__name__)

# =========================
# 1. FEATURE ENGINE
# =========================

def build_features(team_A, team_B, h2h):

    avg_A = sum(team_A) / len(team_A)
    avg_B = sum(team_B) / len(team_B)

    if len(h2h) > 0:
        h2h_A = sum([x[0] for x in h2h]) / len(h2h)
        h2h_B = sum([x[1] for x in h2h]) / len(h2h)
    else:
        h2h_A, h2h_B = avg_A, avg_B

    final_A = (avg_A * 0.6) + (h2h_A * 0.4)
    final_B = (avg_B * 0.6) + (h2h_B * 0.4)

    return final_A, final_B


# =========================
# 2. XGBOOST (simple modèle initial)
# =========================

model = XGBClassifier(eval_metric="logloss")
model.fit([[1.5, 1.2]], [1])

def xgb_predict(A, B):
    proba = model.predict_proba([[A, B]])[0]
    return {"A": proba[1], "B": proba[0]}


# =========================
# 3. POISSON MODEL
# =========================

def poisson_scores(A, B):
    return [
        ("2-1", 0.30),
        ("1-1", 0.25),
        ("2-0", 0.20)
    ]


# =========================
# 4. MONTE CARLO SIMULATION
# =========================

def monte_carlo(A, B, sims=3000):
    A_wins = 0
    B_wins = 0
    draws = 0

    for _ in range(sims):
        sa = np.random.poisson(A)
        sb = np.random.poisson(B)

        if sa > sb:
            A_wins += 1
        elif sb > sa:
            B_wins += 1
        else:
            draws += 1

    return {
        "A": A_wins / sims,
        "B": B_wins / sims,
        "D": draws / sims
    }


# =========================
# 5. FUSION IA
# =========================

def fusion(xgb, monte, A, B):

    final_A = (xgb["A"] * 0.4) + (monte["A"] * 0.3) + (A * 0.3)
    final_B = (xgb["B"] * 0.4) + (monte["B"] * 0.3) + (B * 0.3)

    return final_A, final_B


# =========================
# 6. VALUE BET
# =========================

def value_bet(A, B):
    diff = abs(A - B)

    if diff < 0.05:
        return "NO VALUE"
    elif A > B:
        return "VALUE BET HOME"
    else:
        return "VALUE BET AWAY"


# =========================
# 7. API ROUTE
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    team_A = data["team_A"]
    team_B = data["team_B"]
    h2h = data.get("h2h", [])

    A, B = build_features(team_A, team_B, h2h)

    xgb = xgb_predict(A, B)
    monte = monte_carlo(A, B)

    final_A, final_B = fusion(xgb, monte, A, B)

    total_goals = final_A + final_B

    btts = final_A > 1 and final_B > 1
    over = total_goals > 2.5

    result = {
        "1X2": "Home" if final_A > final_B else "Away" if final_B > final_A else "Draw",
        "BTTS": btts,
        "OverUnder": "Over" if over else "Under",
        "score_top3": poisson_scores(final_A, final_B),
        "value_bet": value_bet(final_A, final_B),
        "prob_home": round(final_A, 2),
        "prob_away": round(final_B, 2)
    }

    return jsonify(result)


# =========================
# 8. RUN SERVER (IMPORTANT FOR RENDER)
# =========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
