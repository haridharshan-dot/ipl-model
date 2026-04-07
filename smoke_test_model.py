import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

base = Path(r"z:/New folder (9)/downloaded")
model = joblib.load(base / "ipl_winner_catboost.joblib")
meta = json.loads((base / "ipl_model_meta.json").read_text(encoding="utf-8"))

feature_cols = meta["feature_cols"]
cat_cols = set(meta["cat_cols"])

row = {
    "season": 2024,
    "city": "Mumbai",
    "venue": "Wankhede Stadium",
    "team1": "Mumbai Indians",
    "team2": "Chennai Super Kings",
    "toss_winner": "Mumbai Indians",
    "toss_decision": "field",
    "dl_applied": 0,
}

df = pd.DataFrame([row])
for col in feature_cols:
    if col not in df.columns:
        df[col] = np.nan

for col in feature_cols:
    if col in cat_cols:
        df[col] = df[col].astype(str).replace("nan", "Unknown").fillna("Unknown")
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df = df[feature_cols]
probs = model.predict_proba(df)[0]
classes = model.classes_
ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:3]

print("SMOKE_TEST_OK")
print("Top-3:", ranked)
