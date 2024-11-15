import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# from src.config import RANDOM_STATE, TEST_SIZE

# Define constants directly
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_and_process_data(filepath):
    """
    Loads data from a CSV file, handling missing values by removing any rows with NaNs.
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)
    return df


def scale_data(data):
    """
    Scales numerical data using the RobustScaler, which is less sensitive to outliers.

    The data.select_dtypes(include=[np.number]) part of the code filters out the
    non-numeric columns from the data DataFrame. This ensures that only numeric data is passed to the RobustScaler.

    include=[np.number] specifies that columns must be of numeric types, which include integers (int) and
    floating points (float). This automatically excludes any non-numeric types such as boolean, categorical, datetime, etc.
    """
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data.select_dtypes(include=[np.number]))
    return scaled_data, scaler


def prepare_data_splits(data_df, target_column="Group", test_size=0.2, random_state=42):
    """
    Splits the data into training and test sets ensuring the test set is kept untouched for final evaluation.
    """
    if target_column not in data_df.columns:
        raise ValueError(f"Missing target column: {target_column}")

    x_train, x_test, y_train, y_test = train_test_split(
        data_df.drop(columns=[target_column]),
        data_df[target_column],
        test_size=test_size,
        stratify=data_df[target_column],
        random_state=random_state,
    )
    train_df = pd.concat([x_train, y_train], axis=1)
    test_df = pd.concat([x_test, y_test], axis=1)
    return train_df, test_df


def process(filepath, target_column="Group", test_size=0.2, random_state=42):
    """
    Process the input data file.

    Args:
        filepath (str): Path to the input data file
        target_column (str): Name of the target column (default: 'Group')
        test_size (float): Proportion of dataset to include in the test split (default: 0.2)

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, object]: Training data, test data, and scaler
    """
    df = load_and_process_data(filepath)
    scaled_data, scaler = scale_data(df.drop(columns=[target_column]))
    df_scaled = pd.DataFrame(
        scaled_data, columns=df.columns[df.columns != target_column], index=df.index
    )
    df_scaled[target_column] = df[target_column]
    # train_df, test_df = prepare_data_splits(
    #     df_scaled, target_column, test_size, random_state
    # )
    # tensor_data = torch.tensor(
    #     train_df.drop(columns=[target_column]).values, dtype=torch.float32
    # )
    # tensor_dataset = TensorDataset(
    #     tensor_data, torch.tensor(train_df[target_column].values)
    # )
    # return tensor_dataset, test_df, scaler
    return df_scaled, scaler 


def get_data_loader(tensor_dataset, batch_size=32):
    """
    Converts TensorDataset to DataLoader.
    """
    loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)
    return loader


# Example of proper usage:
filepath = "/Users/carolkiekhaefer10-2023/Documents/GitHub/gnn-covid-classification/data/data_combined_controls.csv"
train_dataset, test_dataset, scaler = process(filepath, target_column="Group")
train_loader = get_data_loader(train_dataset)


# # If we want to test the module directly
# if __name__ == "__main__":
#     # Test code
#     test_filepath = "/Users/carolkiekhaefer10-2023/Documents/GitHub/gnn-covid-classification/data/data_combined_controls.csv"
#     tensor_dataset, test_df, scaler = process(test_filepath)
#     print("\nProcessing complete!")
#     print(f"Test dataset shape: {test_df.shape}")
