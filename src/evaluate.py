import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, auc,
    average_precision_score
)
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

def evaluate():
    print("Loading data and model...")
    df = load_data(DATA_PATH)
    df = preprocess_data(df)
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # In a real scenario, we should use the same split as training
    # For simplicity here, we'll just split again with same seed
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = FraudDetector.load(MODEL_PATH)
    
    print("Predicting...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pr_curve.png'))
    plt.close()
    
    print(f"Evaluation complete. PR-AUC: {pr_auc:.4f}")

if __name__ == "__main__":
    evaluate()
