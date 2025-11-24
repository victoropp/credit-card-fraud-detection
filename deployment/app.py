from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from model import FraudDetector

app = FastAPI(title="Credit Card Fraud Detection API", description="API for detecting fraudulent transactions")

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgb_fraud_model.pkl')
try:
    model = FraudDetector.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class Transaction(BaseModel):
    features: list

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fraud Detection API"}

@app.post("/predict")
def predict(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert list to numpy array and reshape
        features = np.array(transaction.features).reshape(1, -1)
        
        # Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0, 1]
        
        # Calculate SHAP values
        shap_values = model.get_shap_values(features)[0].tolist()
        
        return {
            "is_fraud": bool(prediction),
            "fraud_probability": float(probability),
            "shap_values": shap_values
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
