from src.label import make_labels
import pandas as pd

raw_path = "data_samples/transactions_sample.csv"
labels_out = "data_samples/labels_60d.csv"

import pandas as pd
tx = pd.read_csv(raw_path, parse_dates=["order_date"])
labels = make_labels(tx, horizon_days=60)
labels.to_csv(labels_out, index=False)
print("Wrote:", labels_out)
