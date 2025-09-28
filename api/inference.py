import joblib, pandas as pd
from functools import lru_cache
from pathlib import Path

FEATURES = [
    "recency_days","orders","monetary","tenure_days",
    "avg_discount","return_rate","category_diversity"
]

@lru_cache
def load_model():
    return joblib.load(Path(__file__).resolve().parent.parent / "models" / "model.joblib")

def predict_one(payload: dict) -> float:
    X = pd.DataFrame([payload], columns=FEATURES).fillna(0)
    model = load_model()
    return float(model.predict_proba(X)[:,1][0])
