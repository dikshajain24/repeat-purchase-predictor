import pandas as pd
import joblib
from pathlib import Path

FEATURES = [
    "recency_days","orders","monetary","tenure_days","avg_discount","return_rate","category_diversity"
]

def score_all(features_csv: str, model_path: str, out_csv: str):
    feats = pd.read_csv(features_csv, parse_dates=["last_order_date","first_order_date"]).fillna(0)
    model = joblib.load(model_path)
    proba = model.predict_proba(feats[FEATURES])[:,1]
    out = feats[["customer_id"]].copy()
    out["proba"] = proba
    out = out.sort_values("proba")
    out["decile"] = pd.qcut(out["proba"].rank(method="first"), 10, labels=list(range(1,11)))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Saved scored CSV → {out_csv}")
