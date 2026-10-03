import json, os
import numpy as np
import xgboost as xgb
from flask import Flask, jsonify, request, send_from_directory

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=None)

booster = xgb.Booster()
booster.load_model(os.path.join(BASE, "model", "cardio_model.json"))
with open(os.path.join(BASE, "model", "meta.json")) as f:
    META = json.load(f)
FEATS = META["features"]
NAMES = [f["name"] for f in FEATS]
MEAN = np.array([f["mean"] for f in FEATS])
SCALE = np.array([f["scale"] for f in FEATS])


@app.get("/")
def index():
    return send_from_directory(os.path.join(BASE, "static"), "index.html")


@app.get("/healthz")
def healthz():
    return "ok"


@app.get("/api/meta")
def meta():
    return jsonify(META)


@app.post("/api/predict")
def predict():
    body = request.get_json(silent=True) or {}
    try:
        x = np.array([float(body[n]) for n in NAMES])
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Provide a numeric value for every marker."), 400
    if not np.isfinite(x).all() or (x < 0).any() or (x > 10000).any():
        return jsonify(error="Values must be finite, non-negative numbers."), 400

    z = (x - MEAN) / SCALE
    dm = xgb.DMatrix(z.reshape(1, -1))
    p = float(booster.predict(dm)[0])
    contrib = booster.predict(dm, pred_contribs=True)[0]  # log-odds; last item = base value
    return jsonify(
        probability=p,
        flagged=p >= 0.5,
        base=float(contrib[-1]),
        logit=float(contrib.sum()),
        features=[dict(name=f["name"], unit=f["unit"], value=float(x[i]), z=float(z[i]),
                       contribution=float(contrib[i]), out_of_range=bool(x[i] < f["lo"] or x[i] > f["hi"]))
                  for i, f in enumerate(FEATS)],
    )


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
