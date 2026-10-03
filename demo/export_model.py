"""One-off: convert the notebook pickles into version-proof files for the web app.
Run from the repo root in your ORIGINAL training env (needs scikit-learn + pandas):
    python demo/export_model.py
"""
import json, os, warnings
import joblib, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NB = os.path.join(ROOT, "jupyter notebook")
OUT = os.path.join(os.path.dirname(__file__), "model")
model = joblib.load(os.path.join(NB, "cardio_model.pkl"))
scaler = joblib.load(os.path.join(NB, "scaler.pkl"))
data = pd.read_csv(os.path.join(ROOT, "Blood_sample", "cardio_data_real.csv"))

NAMES = list(scaler.feature_names_in_)
UNITS = ["mg/dL", "mg/dL", "mg/dL", "mg/dL", "mmHg", "mmHg", "bpm", "kg/m²", "mg/L"]
STEPS = [1, 0.5, 1, 1, 1, 1, 1, 0.1, 0.01]

model.get_booster().save_model(os.path.join(OUT, "cardio_model.json"))

def prob(vals):
    z = scaler.transform(pd.DataFrame([vals], columns=NAMES))
    return float(model.predict_proba(z)[0, 1])

feats = []
for i, n in enumerate(NAMES):
    feats.append(dict(name=n, unit=UNITS[i], step=STEPS[i],
                      mean=float(scaler.mean_[i]), scale=float(scaler.scale_[i]),
                      lo=float(np.floor(max(data[n].min(), 0))), hi=float(np.ceil(data[n].max()))))

def rnd(row):
    return {n: round(float(row[n]) / STEPS[i]) * STEPS[i] if STEPS[i] >= 1 else round(float(row[n]), 2 if STEPS[i] < 0.1 else 1)
            for i, n in enumerate(NAMES)}

presets = []
med = rnd(data[NAMES].median()); presets.append(dict(label="Typical (median) profile", values=med))
for label, ok in (("Example the model does not flag", lambda p: p < 0.05),
                  ("Example the model flags", lambda p: p > 0.99)):
    for _, r in data.iterrows():
        v = rnd(r)
        if ok(prob(v)):
            presets.append(dict(label=label, values=v)); break

json.dump(dict(features=feats, presets=presets), open(os.path.join(OUT, "meta.json"), "w"), indent=1)
for p in presets: print(p["label"], round(prob(p["values"]), 4))
print("exported to", OUT)
