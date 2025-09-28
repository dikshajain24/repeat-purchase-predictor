from src.features import load_raw, build_customer_features
import pandas as pd

raw_path = "data_samples/transactions_sample.csv"
features_out = "data_samples/customer_features.csv"

df = load_raw(raw_path)
feats = build_customer_features(df)
feats.to_csv(features_out, index=False)
print("Wrote:", features_out)
