import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from typing import Tuple, Union, List
from pathlib import Path


class DataPreprocessor:
    """
    Preprocesses data by loading, cleaning, and scaling numeric columns with RobustScaler.

    Attributes:
        scaler (RobustScaler): Scaler used for scaling numeric data to reduce the impact of outliers.
    """

    def __init__(self):
        self.scaler = RobustScaler()

    def load_and_clean_data(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Loads data from a CSV file and removes rows with any NaN values.

        Parameters:
            filepath (Union[str, Path]): The path to the CSV file containing the data.

        Returns:
            pd.DataFrame: A cleaned DataFrame with rows containing NaN values removed.

        Raises:
            FileNotFoundError: If the specified file path does not exist.
            ValueError: If the file is empty or contains no data, or if it cannot be parsed as a CSV.
        """
        filepath = Path(filepath)

        # Check if the file exists
        if not filepath.is_file():
            raise FileNotFoundError(f"The file at {filepath} was not found.")

        # Check for empty file before loading
        if filepath.stat().st_size == 0:
            raise ValueError(f"The file at {filepath} is empty.")

        # Attempt to load the CSV file
        df = pd.read_csv(filepath)

        print(f"Shape of Original Data at loading: {df.shape}")

        # Ensure data is present in the file
        if df.empty:
            raise ValueError(f"The file at {filepath} contains no data.")

        # Drop rows with NaN values
        df.dropna(inplace=True)

        # If a numeric column is incorrectly stored as an object type, we can attempt to convert to float
        for col in df.columns:
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                print(f"INFORMATIONAL Msg: Could not convert {col} to float")

        df.dropna(inplace=True)

        return df
    
    ## NEW FUNCTION TO ENSJRE CONSISTNET SCALING AND DATA HANDLING
    def scale_numeric_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Scales the numeric columns in a DataFrame using RobustScaler.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the data to be scaled.

        Returns:
            np.ndarray: The scaled numeric data as a NumPy array.

        Notes:
            - Only numeric columns are scaled; non-numeric columns are ignored.
            - Missing values in numeric columns are filled with the column mean before scaling.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numeric_cols] = (
            df[numeric_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(df[numeric_cols].mean())
        )
        scaled_data = self.scaler.fit_transform(df[numeric_cols])
        return scaled_data, numeric_cols

    def process(
        self, filepath: Union[str, Path]
    ) -> Tuple[np.ndarray, RobustScaler, pd.DataFrame, List[str]]:
        """
        Full preprocessing pipeline: loads, cleans, and scales data.

        Parameters:
            filepath (Union[str, Path]): The path to the CSV file containing the data.

        Returns:
            Tuple[np.ndarray, RobustScaler, pd.DataFrame]: A tuple containing:
                - The scaled numeric data as a NumPy array.
                - The fitted RobustScaler instance.
                - The cleaned DataFrame (with rows containing NaN values removed).
        """
        original_data = self.load_and_clean_data(filepath)
        scaled_data, col_names = self.scale_numeric_data(original_data)
        return scaled_data, self.scaler, original_data, col_names


def validate_preprocessing_output(
    scaled_data: np.ndarray, scaler: RobustScaler, df: pd.DataFrame
) -> bool:
    """
    Validates the output of the preprocessing pipeline to ensure compatibility.

    Parameters:
        scaled_data (np.ndarray): The scaled data output.
        scaler (RobustScaler): The scaler used for scaling.
        df (pd.DataFrame): The original DataFrame after cleaning.

    Returns:
        bool: True if the scaled data has the same number of rows as the cleaned DataFrame.
    """
    return scaled_data.shape[0] == df.shape[0]


# # Legacy support function
# def process(filepath: str) -> Tuple[np.ndarray, RobustScaler, pd.DataFrame]:
#     """
#     Legacy function for preprocessing data, compatible with previous code versions.

#     Parameters:
#         filepath (str): The path to the CSV file containing the data.

#     Returns:
#         Tuple[np.ndarray, RobustScaler, pd.DataFrame]: Outputs from the DataPreprocessor's process method.
#     """
#     preprocessor = DataPreprocessor()
#     return preprocessor.process(filepath)
