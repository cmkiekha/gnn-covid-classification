# preprocessing_v2.py
import json
from typing import Tuple, Union
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    Handles data preprocessing for COVID-19 data analysis including
    loading, cleaning, scaling, and train/test splitting.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = None
        self.feature_names = None
        self.metadata = {
            "timestamp": datetime.now().isoformat(),
            "random_state": random_state,
            "preprocessing_steps": [],
        }

    def validate_filepath(self, filepath: Union[str, Path]) -> Path:
        """Validate file path and existence."""
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"Data file not found: {filepath}")
            raise FileNotFoundError(f"Data file not found: {filepath}")
        if not filepath.is_file():
            print(f"Path is not a file: {filepath}")
            raise ValueError(f"Path is not a file: {filepath}")
        return filepath

    def validate_dataframe(self, data_df: pd.DataFrame, step: str) -> None:
        """Validate DataFrame at various processing steps."""
        if data_df is None:
            print(f"DataFrame is None at step: {step}")
            raise ValueError(f"DataFrame is None at step: {step}")
        if data_df.empty:
            print(f"Empty DataFrame at step: {step}")
            raise ValueError(f"Empty DataFrame at step: {step}")

    def load_data(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Load and validate data from CSV file."""
        filepath = self.validate_filepath(filepath)
        print(f"Loading data from {filepath}")
        data_df = pd.read_csv(filepath)
        self.validate_dataframe(data_df, "load_data")

        self.metadata["preprocessing_steps"].append(
            {
                "step": "data_loading",
                "timestamp": datetime.now().isoformat(),
                "n_samples": len(data_df),
                "n_features": len(data_df.columns),
            }
        )

        print(f"Loaded {len(data_df)} samples with {len(data_df.columns)} features")
        return data_df

    def save_metadata(self, output_dir: Union[str, Path]) -> None:
        """
        Save preprocessing metadata to file.

        Args:
            output_dir: Directory to save metadata
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = output_dir / "preprocessing_metadata.json"

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
        print("Saved preprocessing metadata to", str(metadata_file))

    def clean_data(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling missing values and invalid entries."""
        self.validate_dataframe(data_df, "clean_data_input")
        initial_samples = len(data_df)

        # Check for and remove NaN values
        nan_counts = data_df.isna().sum()
        if nan_counts.any():
            print("Found NaN values:\n", str(nan_counts[nan_counts > 0]))
            data_df = data_df.dropna()

        # Check for and remove duplicates
        n_duplicates = len(data_df) - len(data_df.drop_duplicates())
        if n_duplicates > 0:
            print("Found", n_duplicates, "duplicate rows")
            data_df = data_df.drop_duplicates()

        self.validate_dataframe(data_df, "clean_data_output")

        self.metadata["preprocessing_steps"].append(
            {
                "step": "data_cleaning",
                "timestamp": datetime.now().isoformat(),
                "initial_samples": initial_samples,
                "final_samples": len(data_df),
                "removed_samples": initial_samples - len(data_df),
                "nan_counts": nan_counts.to_dict(),
                "duplicates_removed": n_duplicates,
            }
        )

        print("Cleaned data:", len(data_df), "samples remaining")
        return data_df

    def validate_split_inputs(self, data_df: pd.DataFrame, target: pd.Series) -> None:
        """Validate inputs for data splitting."""
        if len(data_df) != len(target):
            print("Feature matrix and target vector have different lengths")
            raise ValueError("Feature matrix and target vector have different lengths")
        if target.isna().any():
            print("Target vector contains NaN values")
            raise ValueError("Target vector contains NaN values")

    def prepare_data_splits(
        self, data_df: pd.DataFrame, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Create train/test splits with proper stratification."""
        if not 0 < test_size < 1:
            print(f"Invalid test_size: {test_size}")
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

        # Stratified split ensuring proportional controls
        x_train, x_test, y_train, y_test = train_test_split(
            data_df,
            data_df["target"],
            test_size=test_size,
            stratify=data_df["target"],
            random_state=self.random_state,
        )

        print(f"Split sizes - Train: {len(x_train)}, Test: {len(x_test)}")

        self.metadata["preprocessing_steps"].append(
            {
                "step": "train_test_split",
                "timestamp": datetime.now().isoformat(),
                "train_samples": len(x_train),
                "test_samples": len(x_test),
                "test_size": test_size,
                "stratification": "target",
            }
        )

        return x_train, x_test, y_train, y_test

    def validate_scale_inputs(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> None:
        """Validate inputs for scaling."""
        if train_data.columns.tolist() != test_data.columns.tolist():
            print("Train and test data have different features")
            raise ValueError("Train and test data have different features")

    def scale_data(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scales numerical data using the RobustScaler, which is less sensitive to outliers.

        Args:
            data (pd.DataFrame): The DataFrame containing the data to be scaled.

        Returns:
            numpy.ndarray: The scaled data.
            RobustScaler: The scaler instance that was used to transform the data.
        """
        self.validate_scale_inputs(train_data, test_data)
        self.feature_names = train_data.columns.tolist()

        self.scaler = RobustScaler()
        scaled_train = self.scaler.fit_transform(train_data)
        scaled_test = self.scaler.transform(test_data)

        print(
            "Scaled",
            len(scaled_train),
            "training and",
            len(scaled_test),
            "test samples",
        )

        self.metadata["preprocessing_steps"].append(
            {
                "step": "data_scaling",
                "timestamp": datetime.now().isoformat(),
                "scaling_method": "RobustScaler",
                "n_features": len(self.feature_names),
                "feature_names": self.feature_names,
            }
        )

        return scaled_train, scaled_test

    def create_torch_datasets(
        self,
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[TensorDataset, TensorDataset]:
        """Convert numpy arrays to PyTorch datasets."""
        # Validate shapes
        if x_train.shape[1] != x_test.shape[1]:
            print("Inconsistent feature dimensions between train and test sets")
            raise ValueError(
                "Inconsistent feature dimensions between train and test sets"
            )

        train_tensor_x = torch.FloatTensor(x_train)
        train_tensor_y = torch.FloatTensor(y_train.values)
        test_tensor_x = torch.FloatTensor(x_test)
        test_tensor_y = torch.FloatTensor(y_test.values)

        train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
        test_dataset = TensorDataset(test_tensor_x, test_tensor_y)

        print("Created PyTorch datasets successfully")
        return train_dataset, test_dataset


def process(
    filepath: str, output_dir: str = "results", test_size: float = 0.2
) -> Tuple[TensorDataset, TensorDataset, pd.DataFrame, RobustScaler, int]:
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
    print(f"Starting preprocessing pipeline for {filepath}")

    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Load and clean data
    data_df = preprocessor.load_data(filepath)
    cleaned_data = preprocessor.clean_data(data_df)

    # Validate target column
    if "Group" not in cleaned_data.columns:
        print("Target column 'Group' not found in data")
        raise ValueError("Target column 'Group' not found in data")

    # Stratified split ensuring proportional controls
    x_train, x_test, y_train, y_test = train_test_split(
        cleaned_data,
        cleaned_data["Group"],
        test_size=test_size,
        stratify=cleaned_data["Group"],
        random_state=preprocessor.random_state,
    )

    # Add split information
    x_train["split"] = "development"
    x_test["split"] = "holdout"

    # Scale data
    x_train_scaled, x_test_scaled = preprocessor.scale_data(
        x_train.drop(["Group", "split"], axis=1),
        x_test.drop(["Group", "split"], axis=1),
    )

    # Create PyTorch datasets
    train_dataset, test_dataset = preprocessor.create_torch_datasets(
        x_train_scaled, x_test_scaled, y_train, y_test
    )

    # Save metadata
    preprocessor.save_metadata(output_dir)
    print("Preprocessing completed successfully")

    return (
        train_dataset,
        test_dataset,
        cleaned_data,
        preprocessor.scaler,
        len(preprocessor.feature_names),
    )
