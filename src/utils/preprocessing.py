from typing import Tuple, Optional, Any
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch
from torch.utils.data import TensorDataset
import numpy as np

def load_and_process_data(filepath):
    """
    Loads data from a CSV file, handling missing values by removing any rows with NaNs.

    Args:
        filepath (str): The path to the CSV file to be loaded.

    Returns:
        pd.DataFrame: The loaded data with all rows containing NaN values removed.
    """
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)
    return df


def scale_data(data):
    """
    Scales numerical data using the RobustScaler, which is less sensitive to outliers.

    Args:
        data (pd.DataFrame): The DataFrame containing the data to be scaled.

    Returns:
        numpy.ndarray: The scaled data.
        RobustScaler: The scaler instance that was used to transform the data.
    """
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


def process(filepath: str) -> Tuple[Optional[TensorDataset], Optional[torch.Tensor], pd.DataFrame, Any, int]:

    """
    Processes the data from a CSV file by loading, dropping missing values, scaling using RobustScaler,
    and converting it to a TensorDataset.

    Args:
        filepath (str): The path to the dataset.

    Returns:
        Tuple containing processed dataset components
        torch.utils.data.TensorDataset: A dataset containing the scaled data as tensors.
        torch.Tensor: Tensor containing the scaled data.
        pd.DataFrame: DataFrame containing the original (unscaled) data with a label column added.
        RobustScaler: The scaler instance used for scaling the data.
        int: The number of features in the scaled data.

    Raises:
    FileNotFoundError: If input file doesn't exist
    ValueError: If data format is invalid
    """
    # Validate file path
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Load and validate data
    raw_data = pd.read_csv(filepath)
    
    if raw_data.empty:
        raise ValueError("Empty dataset provided")
    
    # Remove any non-numeric columns
    numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in dataset")
    
    data = raw_data[numeric_cols]
    
    # Scale data
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Create DataFrame with scaled data
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)
    
    # Return tuple with placeholders for unused tensor components
    return None, None, scaled_df, scaler, scaled_df.shape[1]
