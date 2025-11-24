import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import os
from pathlib import Path
from data_loader import load_data, preprocess_data
from model import FraudDetector

# Use relative paths from project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_fraud_model.pkl"
OUTPUT_DIR = PROJECT_ROOT / "notebooks"

def explain():
    print("Loading data and model...")
    df = load_data(DATA_PATH)
    df = preprocess_data(df)
    
    X = df.drop('Class', axis=1)
    
    # Load model
    model_wrapper = FraudDetector.load(MODEL_PATH)
    model = model_wrapper.model
    
    print("Calculating SHAP values (this may take a while)...")
    # Use a sample for speed
    X_sample = X.sample(1000, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    print("Generating plots...")
    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary.png'))
    plt.close()
    
    # Feature importance bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.savefig(os.path.join(OUTPUT_DIR, 'shap_importance.png'))
    plt.close()
    
    print("Explainability analysis complete. Plots saved.")

if __name__ == "__main__":
    explain()
