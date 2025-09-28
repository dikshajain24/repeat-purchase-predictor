import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path

FEATURES = [
    "recency_days","orders","monetary","tenure_days","avg_discount","return_rate","category_diversity"
]

def train_and_save(features_csv: str, labels_csv: str, out_model_path: str, eval_out_csv: str):
    feats = pd.read_csv(features_csv, parse_dates=["last_order_date","first_order_date"])
    labels = pd.read_csv(labels_csv)
    df = feats.merge(labels, on="customer_id", how="inner").fillna(0)

    X = df[FEATURES]
    y = df["label"].astype(int)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(Xtr, ytr)
    ypred = model.predict_proba(Xte)[:,1]

    auc = roc_auc_score(yte, ypred)
    ap  = average_precision_score(yte, ypred)
    print(f"AUC={auc:.3f}  AP={ap:.3f}")

    # save model
    Path(out_model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_model_path)
    print(f"Saved model → {out_model_path}")

    # save eval CSV with probs/deciles for Tableau
    holdout = Xte.copy()
    holdout["customer_id"] = df.loc[Xte.index, "customer_id"].values
    holdout["label"] = yte.values
    holdout["proba"] = ypred
    holdout = holdout.sort_values("proba", ascending=False)
    holdout["decile"] = pd.qcut(holdout["proba"].rank(method="first"), 10, labels=list(range(10,0,-1)))
    Path(eval_out_csv).parent.mkdir(parents=True, exist_ok=True)
    holdout.to_csv(eval_out_csv, index=False)
    print(f"Saved eval CSV → {eval_out_csv}")
