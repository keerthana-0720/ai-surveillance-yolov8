import os

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load model (train.py saves a dict with model + feature_columns)
_model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
_loaded = joblib.load(_model_path)
if isinstance(_loaded, dict) and "model" in _loaded and "feature_columns" in _loaded:
    model = _loaded["model"]
    feature_columns = _loaded["feature_columns"]
else:
    raise RuntimeError(
        "Invalid model format. Please run 'python train.py' again for system metrics."
    )

# Input format
class InputData(BaseModel):
    metrics: dict[str, float]

# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Anomaly Detection API Running"
    }

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    # Build one-row input using the same columns used in training.
    missing = [col for col in feature_columns if col not in data.metrics]
    if missing:
        return {
            "error": f"Missing required metrics: {missing}",
            "required_metrics": feature_columns,
        }

    input_df = pd.DataFrame(
        [[data.metrics[col] for col in feature_columns]],
        columns=feature_columns,
    )

    # Predict
    prediction = model.predict(input_df)

    # Convert result
    if prediction[0] == -1:
        result = "Anomaly"
    else:
        result = "Normal"

    return {
        "input_metrics": data.metrics,
        "prediction": result
    }