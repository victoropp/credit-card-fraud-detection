import pandas as pd
import numpy as np
import os

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads the credit card fraud dataset.
    
    Args:
        filepath (str): Path to the csv file.
        
    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at {filepath}")
    
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic preprocessing for the dataset.
    
    Args:
        df (pd.DataFrame): Raw dataframe.
        
    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """
    # Drop duplicates if any
    df = df.drop_duplicates()
    
    # Basic feature engineering (example)
    # In a real scenario, we would add more complex logic here
    # For now, we just return the dataframe as is or with minimal changes
    return df
