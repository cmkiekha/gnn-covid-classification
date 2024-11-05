import pandas as pd
from sklearn.preprocessing import RobustScaler
import numpy as np
from typing import Tuple

def load_and_process_data(filepath: str) -> pd.DataFrame:
    """
    Loads and cleans the data from a CSV file by removing rows with any NaN values.
    """
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)
    return df

def scale_data(data: pd.DataFrame) -> Tuple[np.ndarray, RobustScaler]:
    """
    Scales the data using RobustScaler, which is less sensitive to outliers.
    """
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data.select_dtypes(include=[np.number]))
    return scaled_data, scaler

def process(filepath: str) -> Tuple[np.ndarray, RobustScaler, pd.DataFrame]:
    """
    Processes the data by loading, cleaning, and scaling.
    """
    df = load_and_process_data(filepath)
    scaled_data, scaler = scale_data(df)
    return scaled_data, scaler, df
