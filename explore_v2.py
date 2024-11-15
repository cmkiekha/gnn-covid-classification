"""
WGAN-GP Training and Evaluation Script for COVID-19 Data Augmentation

This script implements a full pipeline for training a WGAN-GP model to generate synthetic data for COVID-19 analysis,
evaluating the synthetic data quality, and saving both results and evaluation metrics.

Key Functionalities:
1. Environment Setup:
   - Sets random seeds for reproducibility across CPU and GPU.
2. Data Preprocessing and Synthetic Data Generation:
   - Preprocesses original data using scaling and missing value handling.
   - Trains a WGAN-GP model to generate synthetic samples, with adjustable training epochs for debugging or full training.
3. Evaluation:
   - Compares synthetic data to the original dataset using statistical and visual assessments (KS tests, t-SNE, etc.).
   - Saves evaluation plots to a specified directory.
4. Result Saving:
   - Saves generated synthetic data and original data to CSV files.
   - Logs key metrics, including training time and sample counts, to a summary text file.

Key Components:

1. Gradient Penalty (λgp):The lambda_gp parameter (set to 10 in train_wgan_gp) controls the strength of the gradient penalty. 
This penalty enforces Lipschitz continuity, helping the WGAN-GP maintain stability during training by ensuring the critic’s 
gradient norm is close to 1.

2. Critic and Generator Training:
    The critic (or discriminator) evaluates both real and generated (fake) data, computing how close generated 
    samples are to the original distribution.
    The generator tries to improve based on the critic’s feedback to produce more realistic synthetic data.

3. Function train_wgan_gp: This function orchestrates the WGAN-GP training by alternating between updates to the critic and generator. It uses the gradient penalty (lambda_gp) to enforce stability in the critic’s updates, making it a critical part of the WGAN-GP’s design.         

Modules Used:
1. `GAN_v3`: Contains `train_and_generate`, which handles WGAN-GP model training and synthetic data generation.
2. `preprocessing_v4`: Contains `DataPreprocessor`, a class for data loading and preprocessing.
3. `evaluation_v4`: Contains `SyntheticDataEvaluator` for comprehensive evaluation of synthetic data quality.

Configuration:
- Reads configurations from `config.py` for parameters such as file paths, batch size, learning rate, and device selection.

Example Usage:
    To run in debug mode (quick testing with fewer epochs):
        $ python explore.py --mode debug
    To run full training (extended training with more epochs):
        $ python explore.py --mode full

Functions:
    - `initialize_environment()`: Sets random seeds for reproducibility.
    - `generate_synthetic_data(epochs)`: Preprocesses data and generates synthetic samples using WGAN-GP for a specified number of epochs.
    - `create_evaluation_plots(original_data, synthetic_data, numeric_cols)`: Evaluates the synthetic data quality through statistical and visual comparisons.
    - `save_results(original_data, synthetic_data, start_time)`: Saves synthetic data, original data, and evaluation metrics.
    - `main(mode)`: Executes the full pipeline, controlling debug/full training modes.

Raises:
    FileNotFoundError: If the data file specified in `config.DATA_PATH` is not found.
    Exception: For general errors in data generation, evaluation, or saving.

Returns:
    None

Authors:
    CM Kiekhaefer
    Version: 3.0
"""

import warnings
warnings.filterwarnings("ignore")


from datetime import datetime
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Tuple, List
from sklearn.model_selection import KFold

from src.models.data_augmentation.GAN_v3 import (
    train_and_generate,
    train_wgan_gp,
    Generator,
    Critic,
)

from src.config import (
    DATA_PATH,
    RESULT_DIR,
    BATCH_SIZE,
    DEV_EPOCHS,
    DEBUG_EPOCHS,
    RANDOM_STATE,
    CV_N_SPLITS,
    LEARNING_RATE,
    DEVICE,
    SAVE_INFO,
)

from src.preprocessing_v4 import DataPreprocessor
from src.utils.evaluation_v4 import SyntheticDataEvaluator


def initialize_environment():
    """
    Initializes the training environment by setting random seeds for reproducibility.
    Ensures consistent behavior across CPU and GPU, if available.

    This function sets seeds for both PyTorch and NumPy to ensure consistent behavior
    across runs. For CUDA-enabled devices, it also sets a manual seed for CUDA.

    Example Usage:
        initialize_environment()
    """

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_STATE)
        print("Running on GPU")
    else:
        print("Running on CPU")
    print(f"Environment initialized with seed {RANDOM_STATE}")


def process_and_combine_data(
    original_data: pd.DataFrame, synthetic_data: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Combines original and synthetic datasets for comparative evaluation,
    ensuring data type consistency and adding labels to differentiate
    between original and synthetic samples.

    This function is useful for preparing data to assess how well the synthetic
    data represents the original data distribution. It extracts only the numeric
    columns from each dataset, assigns a label to indicate data origin
    ('original' or 'synthetic'), and then combines both datasets.

    Parameters:
        original_data (pd.DataFrame): The original dataset, containing numeric
                                      and possibly non-numeric columns.
        synthetic_data (pd.DataFrame): The synthetic dataset, structured similarly
                                       to the original dataset.

    Returns:
        Tuple[pd.DataFrame, List[str]]:
            - A combined DataFrame of numeric columns from both datasets,
              labeled by data origin.
            - A list of numeric column names retained from the original dataset.

    Example Usage:
        combined_data, numeric_cols = process_and_combine_data(original_df, synthetic_df)

        # `combined_data` has a new 'data_type' column distinguishing original and synthetic rows,
        # while `numeric_cols` lists the numeric features used for evaluation.
    """
    # Extract numeric columns only
    numeric_cols = original_data.select_dtypes(include=[np.number]).columns.tolist()

    # Isolate numeric data from each dataset
    original_numeric = original_data[numeric_cols].copy()
    synthetic_numeric = synthetic_data[numeric_cols].copy()

    # Add identifier column for data source
    original_numeric["data_type"] = "original"
    synthetic_numeric["data_type"] = "synthetic"

    # Combine original and synthetic data into one DataFrame
    combined_data = pd.concat([original_numeric, synthetic_numeric], ignore_index=True)

    return combined_data, numeric_cols


def generate_synthetic_data(epochs: int):
    """
    Generates synthetic data using a WGAN-GP model, with cross-validation
    and data scaling applied.

    Parameters:
        epochs (int): Number of training epochs for the WGAN-GP model.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - Concatenated synthetic data generated across K-Fold splits.
            - Original data after scaling.

    Example Usage:
        synthetic_data, original_data = generate_synthetic_data(epochs=100)

    Notes:
        This function uses K-Fold cross-validation to split the data, train
        a generator and critic model, and generate synthetic data for each fold.
        The generated synthetic samples are then recentered to match the
        distribution of the original training fold.
    """
    print(f"Generating synthetic data with {epochs} epochs")

    # Preprocess data
    print(f"Generating synthetic data with {epochs} epochs")
    preprocessor = DataPreprocessor()
    scaled_data, scaler, original_data, trainable_col_names = preprocessor.process(
        DATA_PATH
    )
    print(f"Original data shape: {original_data.shape}")
    print(f"Scaled data shape: {scaled_data.shape}")

    # Set up KFold cross-validation using CV_N_SPLITS
    kfold = KFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    synthetic_samples = []
    for fold, (train_idx, _) in enumerate(kfold.split(scaled_data)):
        print(f"Processing fold {fold + 1}/{CV_N_SPLITS}")

        # Create DataLoader for the current fold
        train_loader = DataLoader(
            torch.tensor(scaled_data[train_idx], dtype=torch.float32),
            batch_size=BATCH_SIZE,
        )

        # Initialize Generator and Critic
        generator = Generator(scaled_data.shape[1], scaled_data.shape[1]).to(DEVICE)
        critic = Critic(scaled_data.shape[1]).to(DEVICE)

        # Define optimizers
        g_optimizer = Adam(generator.parameters(), lr=LEARNING_RATE)
        c_optimizer = Adam(critic.parameters(), lr=LEARNING_RATE)

        # Train WGAN-GP
        train_wgan_gp(
            train_loader, generator, critic, g_optimizer, c_optimizer, DEVICE, epochs
        )

        # Generate synthetic data for this fold
        latent_samples = torch.randn(
            len(train_loader.dataset), generator.input_dim, device=DEVICE
        )
        synthetic_data = generator(latent_samples).cpu().detach().numpy()

        # Recenter synthetic data to align with original training data distribution
        synthetic_data = SyntheticDataEvaluator.recenter_data(
            synthetic_data, scaled_data[train_idx]
        )

        # Append recentered synthetic data for the fold
        synthetic_samples.append(
            pd.DataFrame(synthetic_data, columns=trainable_col_names)
        )

    # Concatenate synthetic samples across all folds
    return pd.concat(synthetic_samples, ignore_index=True), pd.DataFrame(
        scaled_data, columns=trainable_col_names
    )


def create_evaluation_plots(original_data, synthetic_data, numeric_cols):
    """
    Creates evaluation plots to visually assess the quality of synthetic data.

    Parameters:
        original_data (pd.DataFrame): Original data used for comparison.
        synthetic_data (pd.DataFrame): Synthetic data generated by the model.
        numeric_cols (List[str]): List of numeric column names to evaluate.

    Returns:
        None. Saves evaluation plots to RESULT_DIR as "evaluation_plots.png".

    Example Usage:
        create_evaluation_plots(original_data, synthetic_data, numeric_cols)
    """
    evaluator = SyntheticDataEvaluator()
    results = evaluator.evaluate(original_data, synthetic_data, recenter=True)
    evaluator.plot_evaluation(results)
    plt.savefig(RESULT_DIR / "evaluation_plots.png", dpi=300)


def save_results(original_data, synthetic_data, start_time):
    """
    Saves key metrics and the original and synthetic datasets to CSV files.

    Parameters:
        original_data (pd.DataFrame): Original dataset.
        synthetic_data (pd.DataFrame): Generated synthetic dataset.
        start_time (datetime): The start time of the training process for timing calculations.

    Returns:
        None. Saves metrics and data to RESULT_DIR.

    Example Usage:
        save_results(original_data, synthetic_data, start_time)
    """
    metrics = {
        "TOTAL_SAMPLES_ORIGINAL": len(original_data),
        "TOTAL_SAMPLES_SYNTHETIC": len(synthetic_data),
        "TRAINING_TIME": str(datetime.now() - start_time),
    }
    with open(RESULT_DIR / "results_summary.txt", "w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    if SAVE_INFO:
        original_data.to_csv(
            RESULT_DIR / "original_data.csv", index=False, encoding="utf-8"
        )
        synthetic_data.to_csv(
            RESULT_DIR / "synthetic_data.csv", index=False, encoding="utf-8"
        )


def main(mode="debug"):
    """
    Main function to initialize environment, train WGAN-GP model, generate synthetic data,
    evaluate its quality, and save results.

    Parameters:
        mode (str): Specifies the mode of operation, either 'debug' (for fewer epochs)
                    or 'full' (for full training epochs).

    Example Usage:
        main("debug")  # Runs in debug mode with a limited number of epochs.

    Explanation of ArgumentParser:
        The ArgumentParser is used to:
        - Define acceptable arguments for the script (here, the `--mode` argument).
        - Parse input values provided in the command line.
        - Allow access to specified values within the script (e.g., `args.mode`)
          to control script behavior based on user input.
    """
    start_time = datetime.now()
    print(f"Starting WGAN-GP training in {mode} mode")

    # Initialize environment for reproducibility
    initialize_environment()
    epochs = DEBUG_EPOCHS if mode == "debug" else DEV_EPOCHS

    # Generate synthetic data
    print(f"Generating synthetic data with {epochs} epochs...")
    original_data, synthetic_data = generate_synthetic_data(epochs)

    # Print details of generated data
    print("\nData Inspection:")
    print(f"Shape of original data: {original_data.shape}")
    print(f"Shape of synthetic data: {synthetic_data.shape}")
    print("\nFirst few ORIGINAL records:\n", original_data.head(3))
    print("\nFirst few SYNTHETIC records:\n", synthetic_data.head(3))
                                                          
    # Uncomment these for actual evaluation and result saving after testing
    # numeric_cols = original_data.select_dtypes(include=[np.number]).columns.tolist()
    # create_evaluation_plots(original_data, synthetic_data, numeric_cols)
    # save_results(original_data, synthetic_data, start_time)

    # Combine original and synthetic data for evaluation
    print("\nCombining original and synthetic data for evaluation...")
    combined_data, numeric_cols = process_and_combine_data(
        original_data, synthetic_data
    )
    print(f"Combined data shape: {combined_data.shape}")
    print(f"Count of numeric columns used for evaluation: {len(numeric_cols)}")

    # Evaluate the quality of synthetic data
    print("\nEvaluating synthetic data quality...")
    create_evaluation_plots(original_data, synthetic_data, numeric_cols)

    # Save results to specified directory
    print("\nSaving results and metrics...")
    save_results(original_data, synthetic_data, start_time)

    # Print completion message with timing
    print("Training and evaluation complete.")
    print(f"Total execution time: {datetime.now() - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WGAN-GP training and evaluation")
    parser.add_argument(
        "--mode",
        choices=["debug", "full"],
        default="debug",
        help="Set mode for training: debug or full",
    )
    args = parser.parse_args()
    main(args.mode)
