from src.train import train_and_save

train_and_save(
    features_csv = "data_samples/customer_features.csv",
    labels_csv   = "data_samples/labels_60d.csv",
    out_model_path = "models/model.joblib",
    eval_out_csv = "data_samples/train_eval.csv",
)
