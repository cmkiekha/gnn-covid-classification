"""
The augment_v3.py script is a standalone script that trains the 
WGAN-GP model and generates synthetic data.
"""

from pathlib import Path
import torch

import config
from src.utils.preprocessing import process
from src.models.data_augmentation.GAN import train_and_generate

def main():
    """
    Main execution function to train the WGAN-GP model and generate synthetic data.

    This function loads data from a specified path, processes it, and uses the WGAN-GP model
    to generate synthetic samples. The results are then saved to a CSV file. It allows
    for user interaction to specify the dataset path.

    Arguments:
    None - Uses global config settings and user input for configuration.

    Raises:
    FileNotFoundError: If the dataset path is invalid or the file is not found.
    RuntimeError: If there is an issue with model training or data generation.

    Returns:
    None - Outputs are saved directly to files.
    """
    # Prompt the user for the dataset path or use the default from config
    #dataset_path = input("Enter the dataset path (hit enter to keep default): ")
    #if not dataset_path:
    dataset_path = config.DATA_PATH

    print(f"\nLoading dataset from path: {dataset_path}")

    # Ensure the dataset file exists
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"No file found at the specified path: {dataset_path}")

    # Processing the dataset
    # If you are not using scaled_data and scaler, no need to unpack them
    _, _, _, _, _ = process(dataset_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Start training the WGAN-GP model
    print("\nTraining WGAN-GP model...")
    output = train_and_generate(
        filepath=dataset_path,
        batch_size=config.BATCH_SIZE,
        epochs=config.DEBUG_EPOCHS,
        device=device,
        n_splits=config.CV_N_SPLITS,
        learning_rate=config.LEARNING_RATE
    )

    # Handling the output tuple from train_and_generate
    if isinstance(output, tuple):
        # Assume output is a tuple of DataFrames or similar
        augmented_df = output[0]  # Assuming the first element is what you want to save
        if augmented_df is not None:
            results_path = Path(config.RESULT_DIR) / "augmented_data.csv"
            augmented_df.to_csv(results_path, index=False)
            print(f"Augmented data saved to {results_path}")
    else:
        print("Expected a tuple from the model output, received:", type(output))

if __name__ == "__main__":
    main()
