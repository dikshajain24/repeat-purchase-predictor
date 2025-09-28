from fastapi import FastAPI
from schemas import CustomerFeatures
from inference import predict_one

app = FastAPI(title="Repeat Purchase Predictor (Public)")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(f: CustomerFeatures):
    return {"probability": predict_one(f.model_dump())}
