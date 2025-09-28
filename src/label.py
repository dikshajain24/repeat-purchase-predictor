import pandas as pd

def make_labels(transactions: pd.DataFrame, horizon_days: int = 60) -> pd.DataFrame:
    tx = transactions.copy()
    tx["order_date"] = pd.to_datetime(tx["order_date"])
    max_date = tx["order_date"].max()
    cutoff = max_date - pd.Timedelta(days=horizon_days)

    # last activity per customer BEFORE/ON cutoff
    last_before = (
        tx[tx["order_date"] <= cutoff]
        .groupby("customer_id")["order_date"].max()
        .rename("last_before")
        .reset_index()
    )

    # any purchase AFTER cutoff within horizon
    future = tx[
        (tx["order_date"] > cutoff) &
        (tx["order_date"] <= cutoff + pd.Timedelta(days=horizon_days))
    ][["customer_id"]].drop_duplicates()
    future["label"] = 1

    lab = last_before.merge(future, on="customer_id", how="left").fillna({"label":0})
    return lab[["customer_id","label"]]
