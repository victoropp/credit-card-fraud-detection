import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, auc,
    average_precision_score, accuracy_score, f1_score
)
import joblib
import os
import json
from pathlib import Path
from data_loader import load_data, preprocess_data
from model import FraudDetector

# Use relative paths from project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_fraud_model.pkl"
RESULTS_DIR = PROJECT_ROOT / "results"

def generate_metrics():
    print("Loading data and model...")
    df = load_data(DATA_PATH)
    df = preprocess_data(df)
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split data (same seed as training)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model_wrapper = FraudDetector.load(MODEL_PATH)
    model = model_wrapper.model
    
    # 1. Training History (Loss Curve)
    # Note: This requires the model to have been trained with eval_set. 
    # If loaded from file, history might be lost unless we retrain or saved it separately.
    # Since we just updated train.py, let's assume we might need to retrain or just check if results are available.
    # For now, we will try to access evals_result() if available.
    try:
        results = model.evals_result()
        epochs = len(results['validation_0']['logloss'])
        x_axis = range(0, epochs)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
        plt.plot(x_axis, results['validation_1']['logloss'], label='Test')
        plt.legend()
        plt.ylabel('Log Loss')
        plt.xlabel('Epochs')
        plt.title('XGBoost Training Loss')
        plt.savefig(os.path.join(RESULTS_DIR, 'training_loss.png'))
        plt.close()
        print("Training loss chart generated.")
    except Exception as e:
        print(f"Could not generate training loss chart (history might not be saved in model object): {e}")
    
    # 2. Evaluation Metrics (Train vs Test)
    print("Calculating metrics...")
    metrics = {}
    
    for name, X_set, y_set in [('Train', X_train, y_train), ('Test', X_test, y_test)]:
        y_pred = model.predict(X_set)
        y_prob = model.predict_proba(X_set)[:, 1]
        
        metrics[name] = {
            'Accuracy': accuracy_score(y_set, y_pred),
            'F1 Score': f1_score(y_set, y_pred),
            'PR AUC': average_precision_score(y_set, y_prob),
            'ROC AUC': 0.0 # Placeholder, calculated below
        }
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_set, y_prob)
        roc_auc = auc(fpr, tpr)
        metrics[name]['ROC AUC'] = roc_auc
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{name} ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(RESULTS_DIR, f'{name.lower()}_roc_curve.png'))
        plt.close()
        
        # PR Curve
        precision, recall, _ = precision_recall_curve(y_set, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR curve (area = {metrics[name]["PR AUC"]:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{name} Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(RESULTS_DIR, f'{name.lower()}_pr_curve.png'))
        plt.close()
        
        # Confusion Matrix
        cm = confusion_matrix(y_set, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(RESULTS_DIR, f'{name.lower()}_confusion_matrix.png'))
        plt.close()

    # Save metrics to JSON
    with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Save metrics to text file for easy reading
    with open(os.path.join(RESULTS_DIR, 'metrics_report.txt'), 'w') as f:
        f.write("Model Evaluation Metrics\n")
        f.write("========================\n\n")
        for name, values in metrics.items():
            f.write(f"{name} Set:\n")
            for metric, value in values.items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")
            
    print(f"Metrics and charts saved to {RESULTS_DIR}")

if __name__ == "__main__":
    generate_metrics()
