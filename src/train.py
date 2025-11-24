import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, average_precision_score
import os
from pathlib import Path
from data_loader import load_data, preprocess_data
from model import FraudDetector

# Use relative paths from project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_fraud_model.pkl"

def train():
    print("Loading data...")
    df = load_data(DATA_PATH)
    df = preprocess_data(df)
    
    # Prepare features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Calculate scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    # Train model
    print("Training model...")
    model = FraudDetector(scale_pos_weight=scale_pos_weight, n_estimators=200)
    
    # Use eval_set to track performance on train and test
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    
    # Evaluate
    print("Evaluating...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred))
    print(f"PR-AUC: {average_precision_score(y_test, y_pred_proba):.4f}")
    
    # Save model
    print(f"Saving model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    print("Done.")

if __name__ == "__main__":
    train()
