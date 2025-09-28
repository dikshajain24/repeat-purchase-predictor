from src.score import score_all
score_all(
    features_csv = "data_samples/customer_features.csv",
    model_path   = "models/model.joblib",
    out_csv      = "data_samples/gold_repeat_predictions.csv",
)
