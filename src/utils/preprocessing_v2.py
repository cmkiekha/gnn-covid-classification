
# preprocessing_v2.py
#from src.models.data_augmentation.GAN_v2 import COVIDDataAugmentation
from typing import Tuple, Union
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
            'timestamp': datetime.now().isoformat(),
            'random_state': random_state,
            'preprocessing_steps': []
        }

    def validate_filepath(self, filepath: Union[str, Path]) -> Path:
        """Validate file path and existence."""
        filepath = Path(filepath)
        if not filepath.exists():
            logger.error(f"Data file not found: {filepath}")
            raise FileNotFoundError(f"Data file not found: {filepath}")
        if not filepath.is_file():
            logger.error(f"Path is not a file: {filepath}")
            raise ValueError(f"Path is not a file: {filepath}")
        return filepath

    def validate_dataframe(self, df: pd.DataFrame, step: str) -> None:
        """Validate DataFrame at various processing steps."""
        if df is None:
            logger.error(f"DataFrame is None at step: {step}")
            raise ValueError(f"DataFrame is None at step: {step}")
        if df.empty:
            logger.error(f"Empty DataFrame at step: {step}")
            raise ValueError(f"Empty DataFrame at step: {step}")

    def load_data(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Load and validate data from CSV file."""
        filepath = self.validate_filepath(filepath)
        logger.info(f"Loading data from {filepath}")
        
        df = pd.read_csv(filepath)
        self.validate_dataframe(df, "load_data")
        
        self.metadata['preprocessing_steps'].append({
            'step': 'data_loading',
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(df),
            'n_features': len(df.columns)
        })
        
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")

    def save_metadata(self, output_dir: Union[str, Path]) -> None:
        """
        Save preprocessing metadata to file.
        
        Args:
            output_dir: Directory to save metadata
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_file = output_dir / 'preprocessing_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info("Saved preprocessing metadata to %s", str(metadata_file))   
        

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling missing values and invalid entries."""
        self.validate_dataframe(df, "clean_data_input")
        initial_samples = len(df)
        
        # Check for and remove NaN values
        nan_counts = df.isna().sum()
        if nan_counts.any():
            logger.warning("Found NaN values:\n%s", 
                         str(nan_counts[nan_counts > 0]))
        df = df.dropna()
        
        # Check for and remove duplicates
        n_duplicates = len(df) - len(df.drop_duplicates())
        if n_duplicates > 0:
            logger.warning("Found %d duplicate rows", n_duplicates)
        df = df.drop_duplicates()
        
        self.validate_dataframe(df, "clean_data_output")
        
        self.metadata['preprocessing_steps'].append({
            'step': 'data_cleaning',
            'timestamp': datetime.now().isoformat(),
            'initial_samples': initial_samples,
            'final_samples': len(df),
            'removed_samples': initial_samples - len(df),
            'nan_counts': nan_counts.to_dict(),
            'duplicates_removed': n_duplicates
        })
        
        logger.info("Cleaned data: %d samples remaining", len(df))
        return df

    def validate_split_inputs(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Validate inputs for data splitting."""
        if len(X) != len(y):
            logger.error("Feature matrix and target vector have different lengths")
            raise ValueError("Feature matrix and target vector have different lengths")
        if y.isna().any():
            logger.error("Target vector contains NaN values")
            raise ValueError("Target vector contains NaN values")

    def prepare_data_splits(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Create train/test splits with proper stratification."""
        self.validate_split_inputs(X, y)
        
        if not 0 < test_size < 1:
            logger.error(f"Invalid test_size: {test_size}")
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=self.random_state
        )
        
        logger.info(f"Split sizes - Train: {len(X_train)}, Test: {len(X_test)}")
        
        self.metadata['preprocessing_steps'].append({
            'step': 'train_test_split',
            'timestamp': datetime.now().isoformat(),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'test_size': test_size,
            'stratification': 'target'
        })
        
        return X_train, X_test, y_train, y_test

    def validate_scale_inputs(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
        """Validate inputs for scaling."""
        if train_data.columns.tolist() != test_data.columns.tolist():
            logger.error("Train and test data have different features")
            raise ValueError("Train and test data have different features")

    def scale_data(
        self, 
        train_data: pd.DataFrame, 
        test_data: pd.DataFrame
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
        
        logger.info("Scaled %d training and %d test samples",
                   len(scaled_train), len(scaled_test))
        
        self.metadata['preprocessing_steps'].append({
            'step': 'data_scaling',
            'timestamp': datetime.now().isoformat(),
            'scaling_method': 'RobustScaler',
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names
        })
        
        return scaled_train, scaled_test

    def create_torch_datasets(
        self, 
        X_train: np.ndarray, 
        X_test: np.ndarray, 
        y_train: np.ndarray, 
        y_test: np.ndarray
    ) -> Tuple[TensorDataset, TensorDataset]:
        """Convert numpy arrays to PyTorch datasets."""
        # Validate shapes
        if X_train.shape[1] != X_test.shape[1]:
            logger.error("Inconsistent feature dimensions between train and test sets")
            raise ValueError("Inconsistent feature dimensions between train and test sets")
        
        train_tensor_x = torch.FloatTensor(X_train)
        train_tensor_y = torch.FloatTensor(y_train.values)
        test_tensor_x = torch.FloatTensor(X_test)
        test_tensor_y = torch.FloatTensor(y_test.values)
        
        train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
        test_dataset = TensorDataset(test_tensor_x, test_tensor_y)
        
        logger.info("Created PyTorch datasets successfully")
        return train_dataset, test_dataset

def process(
    filepath: str,
    output_dir: str = "results",
    test_size: float = 0.2
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
    logger.info(f"Starting preprocessing pipeline for {filepath}")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    # Load and clean data
    data = preprocessor.load_data(filepath)

    cleaned_data = preprocessor.clean_data(data)

    # Validate target column
    if 'target' not in cleaned_data.columns:
        logger.error("Target column 'target' not found in data")
        raise ValueError("Target column 'target' not found in data")
    
    # Load and clean data
    
    if 'target' not in cleaned_data.columns:
        logger.error("Target column 'target' not found in data")
        raise ValueError("Target column 'target' not found in data")
    
    # Separate features and target
    X = cleaned_data.drop('target', axis=1)
    y = cleaned_data['target']
    
    # Create train/test splits
    X_train, X_test, y_train, y_test = preprocessor.prepare_data_splits(
        X, y, test_size=test_size
    )
    
    # Scale data
    X_train_scaled, X_test_scaled = preprocessor.scale_data(X_train, X_test)
    
    # Create PyTorch datasets
    train_dataset, test_dataset = preprocessor.create_torch_datasets(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # Save metadata
    preprocessor.save_metadata(output_dir)
    
    logger.info("Preprocessing completed successfully")
    return (
        train_dataset,
        test_dataset,
        cleaned_data,
        preprocessor.scaler,
        len(preprocessor.feature_names)
    )

if __name__ == "__main__":
    # Validate environment
    required_packages = ['matplotlib', 'seaborn', 'pandas', 'numpy', 'torch', 'sklearn']
    missing_packages = []
    
    for package in required_packages:
        if package not in globals():
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install required packages:")
        print(f"pip install {' '.join(missing_packages)}")
    else:
        # Process data
        result = process("data/data_combined_controls.csv")
        logger.info("Script completed successfully")

