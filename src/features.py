import pandas as pd
import numpy as np

REQUIRED_COLS = ["customer_id","order_id","order_date","qty","price","discount","is_return","channel","category"]

def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["order_date"])
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df

def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()
    g["gmv"] = g["qty"] * g["price"]
    agg = (
        g.groupby("customer_id")
         .agg(
             last_order_date = ("order_date","max"),
             first_order_date= ("order_date","min"),
             orders          = ("order_id","nunique"),
             monetary        = ("gmv","sum"),
             avg_discount    = ("discount","mean"),
             returns         = ("is_return","sum"),
             lines           = ("order_id","count"),
             category_diversity=("category","nunique"),
         )
         .reset_index()
    )
    max_dt = g["order_date"].max()
    agg["recency_days"] = (max_dt - agg["last_order_date"]).dt.days
    agg["tenure_days"]  = (agg["last_order_date"] - agg["first_order_date"]).dt.days
    agg["return_rate"]  = np.where(agg["lines"]>0, agg["returns"]/agg["lines"], 0.0)
    return agg.drop(columns=["returns","lines"])
