
# path tp GAN_v3.py
from src.models.data_augmentation.GAN_v3 import train_and_generate
#path to DataPreprocessor
from src.preprocessing_v4 import DataPreprocessor
import config
from pathlib import Path

def main():
    """
    Main function to preprocess data, generate synthetic samples using WGAN-GP, 
    and save the generated data to a CSV file.

    Workflow:
    1. Loads and preprocesses the original data from the file path specified in the config.
       - Missing values are handled, and numeric data is scaled.
    2. Generates synthetic data using the WGAN-GP model.
       - Uses the scaled data as input for the generator, with the number of epochs defined in config.DEBUG_EPOCHS.
    3. Saves the generated synthetic data to a CSV file in the output directory specified in config.RESULT_DIR.

    Files:
        - Input data is loaded from config.DATA_PATH.
        - Synthetic data is saved to config.RESULT_DIR/augmented_data.csv.

    Raises:
        FileNotFoundError: If the specified data file in config.DATA_PATH does not exist.
        Exception: If the synthetic data generation or saving fails.

    Example:
        To run the function as a script:
        $ python script_name.py

    Returns:
        None
    """
    scaled_data, scaler, original_data = DataPreprocessor().process(config.DATA_PATH)
    synthetic_data, _ = train_and_generate(scaled_data, original_data, config.DEBUG_EPOCHS, scaler)
    result_path = Path(config.RESULT_DIR) / "augmented_data.csv"
    synthetic_data.to_csv(result_path, index=False)
    print(f"Synthetic data saved to {result_path}")

if __name__ == "__main__":
    main()
