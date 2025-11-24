import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from data_loader import load_data, preprocess_data

# Set style
sns.set(style="whitegrid")

# Use relative paths from project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
OUTPUT_DIR = PROJECT_ROOT / "notebooks"

def perform_eda():
    print("Loading data...")
    df = load_data(DATA_PATH)
    df = preprocess_data(df)
    
    print(f"Data shape: {df.shape}")
    print("Missing values:")
    print(df.isnull().sum().max())
    
    print("Class distribution:")
    print(df['Class'].value_counts(normalize=True))
    
    # 1. Class Imbalance Plot
    try:
        plt.figure(figsize=(6, 4))
        ax = sns.countplot(x='Class', data=df)
        plt.title('Class Distribution (0: Normal, 1: Fraud)')
        plt.yscale('log')
        
        # Add labels
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom')
                        
        plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'))
        plt.close()
    except Exception as e:
        print(f"Error plotting class distribution: {e}")
    
    # 2. Time Distribution
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='Time', hue='Class', bins=50, kde=True, common_norm=False, stat='density')
        plt.title('Transaction Time Distribution by Class')
        plt.savefig(os.path.join(OUTPUT_DIR, 'time_distribution.png'))
        plt.close()
    except Exception as e:
        print(f"Error plotting time distribution: {e}")
    
    # 3. Amount Distribution (Log scale)
    try:
        plt.figure(figsize=(10, 6))
        # Add small constant to avoid log(0)
        df['Amount_Log'] = np.log1p(df['Amount'])
        sns.histplot(data=df, x='Amount_Log', hue='Class', bins=50, kde=True, common_norm=False, stat='density')
        plt.title('Transaction Amount Distribution by Class (Log Scale)')
        plt.savefig(os.path.join(OUTPUT_DIR, 'amount_distribution.png'))
        plt.close()
    except Exception as e:
        print(f"Error plotting amount distribution: {e}")
    
    # 4. Correlation Matrix
    try:
        plt.figure(figsize=(12, 10))
        corr = df.corr()
        sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size': 20})
        plt.title('Correlation Matrix')
        plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'))
        plt.close()
    except Exception as e:
        print(f"Error plotting correlation matrix: {e}")
    
    print("EDA complete. Plots saved to notebooks directory.")

if __name__ == "__main__":
    perform_eda()
