# Cell 1: Documentation and Imports
"""
WGAN-GP Training and Evaluation Notebook for COVID-19 Control Data Augmentation

This notebook implements a Wasserstein GAN with Gradient Penalty (WGAN-GP) for generating
synthetic COVID-19 control samples. The implementation focuses on data quality and validation.

Key Components:
1. Data Loading and Preprocessing
   - Target column identification ('group' or 'target')
   - Data validation and cleaning
   - Feature scaling and normalization

2. WGAN-GP Training with k-fold Cross Validation
   - K-fold validation (k=3) for robust evaluation
   - Gradient penalty for Wasserstein distance
   - Dynamic batch processing

3. Synthetic Data Evaluation
   - Statistical distribution matching
   - Feature-wise comparisons
   - Quality metrics and visualization

4. Analysis and Reporting
   - Comprehensive quality metrics
   - Visual comparisons
   - Detailed reporting

Author: CM Kiekhaefer
Date: 11-02-2024
Version: 1.0
"""

# Standard library imports
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
from collections import namedtuple


# Define core data structure
DataInfo = namedtuple(
    "DataInfo",
    [
        "dataset",  # Original dataset
        "tensor_data",  # Data in tensor format
        "scaled_data",  # Scaled data
        "scaler",  # Fitted scaler object
        "n_features",  # Number of features
    ],
)


# Cell 2: Environment Setup and Verification
def verify_environment():
    """
    Verify all required package installations and versions.

    Checks:
    1. Core packages: numpy, pandas, torch
    2. Visualization: matplotlib, seaborn
    3. Machine learning: scikit-learn
    4. Support packages: tqdm, ipykernel

    Returns:
        bool: True if all required packages are available
    """
    print("Environment Verification")
    print("-" * 50)

    packages = {
        "numpy": "NumPy",
        "pandas": "Pandas",
        "matplotlib": "Plotting",
        "seaborn": "Statistical Visualization",
        "scipy": "Scientific Computing",
        "sklearn": "Machine Learning",
        "torch": "PyTorch",
        "tqdm": "Progress Bars",
        "ipykernel": "Jupyter Support",
    }

    all_passed = True
    for package, description in packages.items():
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            print(f"✓ {description:<25} ({package:<12} version: {version})")
        except ImportError as e:
            all_passed = False
            print(f"✗ {description:<25} ({package:<12} ERROR: {str(e)})")

    print(f"\nPython version: {sys.version.split()[0]}")
    return all_passed


# Cell 3: Package Imports and Setup (only if verification passes)
if True:  # verify_environment():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch

    # Import local modules
    import config
    from src.models.data_augmentation.GAN_v2 import train_and_generate
    from src.utils.preprocessing_v2 import process
    from src.utils.evaluation_v2 import SyntheticDataEvaluator

    # Set plotting configurations
    def setup_plotting():
        """Configure plotting settings."""
        sns.set_theme(style="whitegrid")
        plt.rcParams.update(
            {
                "figure.figsize": (12, 8),
                "figure.dpi": 300,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "lines.linewidth": 2,
                "grid.alpha": 0.3,
            }
        )
        sns.set_palette("husl")

        # Verify plotting setup
        plt.figure()
        sns.lineplot(x=[1, 2, 3], y=[1, 2, 3])
        plt.title("Plotting Test")
        plt.close()
        print("✓ Plotting configuration successful")

    # Initialize plotting
    setup_plotting()

    # Set random seeds
    torch.manual_seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("\nEnvironment setup complete. Ready to proceed with analysis.")
else:
    raise EnvironmentError("Please fix package installation issues before proceeding.")


# Cell 4: Configuration and Data Processing Functions
def print_config(local_config: Dict[str, Any]) -> None:
    """
    Display configuration settings with clear parameter sources.

    Args:
        config: Configuration dictionary containing all parameters
    """
    print("\nConfiguration Settings:")
    print("-" * 50)
    print(f"Data path: {local_config['data_path']}")
    print(f"Results directory: {local_config['results_dir']}")

    print("\nModel Parameters:")
    for key, value in local_config["model_params"].items():
        print(f"  {key}: {value}")

    print("\nVisualization Parameters:")
    for key, value in local_config["visualization_params"].items():
        print(f"  {key}: {value}")


# Cell 4: Configuration
def setup_config(is_debug: bool = True) -> Dict[str, Any]:
    """
    Initialize and validate configuration settings.

    Creates a configuration dictionary with:
    1. Data paths and directories
    2. Model parameters (batch size, epochs, etc.)
    3. Cross-validation settings
    4. Visualization parameters

    Returns:
        Dict containing validated configuration

    Raises:
        AssertionError: If configuration validation fails
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(config.RESULT_DIR) / timestamp

    loaded_config = {
        "data_path": config.DATA_PATH,
        "results_dir": results_dir,
        "model_params": {
            "batch_size": config.BATCH_SIZE,
            "epochs": config.DEBUG_EPOCHS,
            # "random_state": config.RANDOM_STATE,
            "n_splits": config.CV_N_SPLITS,
            "device": config.DEVICE,
        },
        "visualization_params": {"figsize": (12, 8), "dpi": 300, "style": "whitegrid"},
    }

    if not is_debug:
        print("Using DEV configs")
        loaded_config["model_params"]["epochs"] = config.DEV_EPOCHS
    else:
        print("Using DEBUG configs")

    results_dir.mkdir(parents=True, exist_ok=True)
    # Validate configuration
    assert (
        loaded_config["model_params"]["batch_size"] == config.BATCH_SIZE
    ), "Batch size mismatch"
    assert (
        loaded_config["model_params"]["n_splits"] == config.CV_N_SPLITS
    ), "CV splits mismatch"
    assert loaded_config["model_params"]["device"] == config.DEVICE, "Device mismatch"

    return loaded_config


# Cell 5: Data Processing
def inspect_data(filepath: str) -> list:
    """
        Inspect and validate input data file.

        Performs:
        1. File existence check
        2. Column inspection
        3. Target column identification
        4. Data value validation

        Args:
            filepath: Path to data file

    Returns:
            list: Available column names

    Notes:
            Handles both 'group' and 'target' as valid target columns
    """
    # Verify file exists
    if not Path(filepath).exists():
        print(f"Error: File not found at {filepath}")
        return None

    # Read data
    print("\nReading data...")
    df = pd.read_csv(filepath)

    print("\nData Inspection:")
    print("-" * 50)
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumns found:")
    for col in df.columns:
        print(f"  - {col}")

    # Check for potential target columns
    potential_target_cols = [
        col
        for col in df.columns
        if col.lower() in ["target", "group", "label", "class", "status"]
    ]

    if potential_target_cols:
        print("\nPotential target columns found:")
        for col in potential_target_cols:
            print(f"  - {col}")
            if col in df.columns:
                print(f"    Values: {df[col].unique()}")
    else:
        print("\nNo obvious target column found")

    return df.columns.tolist()


def validate_and_process_data(filepath: str) -> Tuple[pd.DataFrame, str]:
    """
    Validate data and identify target column.

    Args:
        filepath: Path to the data file
    Returns:
        Tuple of (DataFrame, target_column_name)
    """
    columns = inspect_data(filepath)
    assert columns is not None, "Failed to read data file"

    # Read data
    df = pd.read_csv(filepath)

    # First check for 'group' column
    if "Group" in df.columns:
        target_col = "Group"
        print("\nUsing 'Group' as target column")
    # Then check for 'target' column
    elif "target" in df.columns:
        target_col = "target"
        print("\nUsing 'target' as target column")
    else:
        raise ValueError("Group and target not found!")

    # Verify target column values
    unique_values = df[target_col].unique()
    assert (
        len(unique_values) == 1 and unique_values[0] == 0
    ), f"Target column must be all 0 since this is control, found values: {unique_values}"

    # Rename target column if needed
    if target_col != "target":
        df = df.rename(columns={target_col: "target"})
        print(f"\nRenamed column '{target_col}' to 'target'")

    return df, target_col


def load_and_process_data(local_config: Dict[str, Any]) -> DataInfo:
    """Load and preprocess the data."""
    print("\nLoading and processing data...")

    # Validate and get data with correct target column
    _, original_target_col = validate_and_process_data(local_config["data_path"])

    # Process data
    dataset, tensor_data, scaled_data, scaler, n_features = process(
        local_config["data_path"]
    )

    # Validate processed data
    # validate_data(scaled_data)

    # Print information
    print("\nDataset Information:")
    print(f"Original shape: {scaled_data.shape}")
    print(f"Number of features: {n_features}")
    print(f"Original target column: '{original_target_col}' (renamed to 'target')")
    print(f"Control samples: {(scaled_data['Group'] == 0).sum()}")
    print(f"COVID-19 cases: {(scaled_data['Group'] == 1).sum()}")

    return DataInfo(dataset, tensor_data, scaled_data, scaler, n_features)


# def validate_synthetic_data(
#     synthetic_data: pd.DataFrame, original_data: pd.DataFrame
# ) -> None:
#     """Validate synthetic data requirements."""
#     assert len(synthetic_data) > 0, "No synthetic samples generated"
#     assert synthetic_data.shape[1] == original_data.shape[1], "Feature mismatch"
#     assert not synthetic_data.isnull().any().any(), "Synthetic data contains NaN values"

#     # Check value ranges
#     for col in synthetic_data.columns:
#         if col not in ["data_type", "fold", "sample_id", "target", "split"]:
#             orig_range = original_data[col].max() - original_data[col].min()
#             synth_range = synthetic_data[col].max() - synthetic_data[col].min()
#             assert (
#                 abs(orig_range - synth_range) / orig_range < 0.5


def train_and_evaluate_model(
    local_config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Train WGAN-GP and evaluate synthetic data, with real-time loss visualization.

    Args:
        config: Configuration dictionary containing model parameters and paths

    Returns:
        Tuple containing:
        - synthetic_data: Generated synthetic samples
        - original_data: Original training data
        - results: Evaluation metrics and results
    """
    print("\nTraining WGAN-GP and generating synthetic samples...")

    # Set up the plotting environment
    plt.ion()  # Turn on interactive mode for live updates
    _, ax = plt.subplots()
    (g_loss_plt,) = ax.plot([], [], label="Generator Loss")
    (d_loss_plt,) = ax.plot([], [], label="Discriminator Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()

    # Initialize lists to store loss values
    g_losses, d_losses = [], []

    # Assume the number of epochs is part of the configuration
    num_epochs = local_config["model_params"]["epochs"]

    for epoch in range(num_epochs):
        # Train model and generate data for one epoch
        g_loss, d_loss = train_and_generate(
            filepath=local_config["data_path"],
            save_info=True,
            **local_config["model_params"],
        )

        if epoch % 5:
            print(f"Epoch {epoch + 1} -- G Loss: {g_loss} -- D Loss: {d_loss}")

        # Append the losses for plotting
        g_losses.append(g_loss)
        d_losses.append(d_loss)

        # Update the plot data
        g_loss_plt.set_data(range(len(g_losses)), g_losses)
        d_loss_plt.set_data(range(len(d_losses)), d_losses)

        # # Adjust plot limits
        # ax.relim()  # Recalculate limits
        # ax.autoscale_view(True, True, True)  # Rescale the view based on the limits

        # Draw and pause to update the plot
        plt.draw()
        plt.pause(0.1)  # Short pause to allow plot updates

    # Disable interactive mode once training is complete to finalize the plot
    plt.ioff()

    # Validate synthetic data
    # validate_synthetic_data(g_losses, d_losses)

    # Initialize evaluator
    evaluator = SyntheticDataEvaluator(output_dir=local_config["results_dir"])
    # Initialize evaluator
    evaluator = SyntheticDataEvaluator(output_dir=local_config["results_dir"])

    # WRONG LOGIC ---------------- The below is passing losses, it should be samples
    # Each data type is a pd.DataFrame
    results = evaluator.evaluate_synthetic_data(
        original_data=g_losses,  # Example usage; adjust as needed
        synthetic_data=d_losses,  # Example usage; adjust as needed
        recenter=True,
    )

    # Add original and synthetic data to results
    results["original_data"] = g_losses  # Example usage; adjust as needed
    results["synthetic_data"] = d_losses  # Example usage; adjust as needed

    return results


def plot_evaluation_metrics(
    local_config: Dict[str, Any], results: Dict[str, Any]
) -> None:
    """Generate and save evaluation plots."""
    print("\nGenerating evaluation visualizations...")

    # Ensure results contain required keys
    required_keys = ["distribution_comparison", "synthetic_data", "original_data"]
    assert all(key in results for key in required_keys), "Missing required results"

    # KS Statistics Distribution
    plt.figure(figsize=local_config["visualization_params"]["figsize"])
    sns.histplot(
        data=results["distribution_comparison"], x="KS_Statistic", kde=True, bins=20
    )
    plt.title("Distribution of KS Statistics")
    plt.xlabel("KS Statistic")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(local_config["results_dir"] / "ks_statistics_distribution.png", dpi=300)
    plt.close()

    # Feature-wise Comparisons
    recentered_synthetic = results["synthetic_data"]
    original_data = results["original_data"]

    feature_cols = [
        col
        for col in original_data.columns
        if col not in ["data_type", "fold", "sample_id", "target", "split"]
    ]

    for column in feature_cols:
        plt.figure(figsize=local_config["visualization_params"]["figsize"])

        # Plot distributions
        sns.kdeplot(data=original_data[column], label="Original", alpha=0.7)
        sns.kdeplot(
            data=recentered_synthetic[column], label="Synthetic (Recentered)", alpha=0.7
        )

        # Add statistics
        orig_mean = original_data[column].mean()
        synth_mean = recentered_synthetic[column].mean()

        plt.axvline(
            orig_mean,
            color="blue",
            linestyle="--",
            alpha=0.5,
            label=f"Original Mean: {orig_mean:.2f}",
        )
        plt.axvline(
            synth_mean,
            color="orange",
            linestyle="--",
            alpha=0.5,
            label=f"Synthetic Mean: {synth_mean:.2f}",
        )

        plt.title(f"Distribution Comparison: {column}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)

        # Save plot
        plt.savefig(
            local_config["results_dir"] / f"feature_distribution_{column}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def generate_summary_report(
    local_config: Dict[str, Any], results: Dict[str, Any]
) -> None:
    """Generate and save comprehensive evaluation report."""
    # Validate results
    assert (
        "distribution_comparison" in results
    ), "Missing distribution comparison results"

    report = {
        "Evaluation Summary": {
            "Original Samples": len(results["original_data"]),
            "Synthetic Samples": len(results["synthetic_data"]),
            "Features": len(results["original_data"].columns),
            "Recentering Applied": results["recentering_applied"],
        },
        "Quality Metrics": {
            "Mean KS Statistic": results["distribution_comparison"][
                "KS_Statistic"
            ].mean(),
            "Median KS Statistic": results["distribution_comparison"][
                "KS_Statistic"
            ].median(),
            "Features with Good Match (KS < 0.1)": (
                results["distribution_comparison"]["KS_Statistic"] < 0.1
            ).sum(),
            "Percentage of Well-matched Features": f"{(results['distribution_comparison']['KS_Statistic'] < 0.1).mean()*100:.2f}%",
        },
        "Training Parameters": local_config["model_params"],
    }

    # Save report
    report_path = local_config["results_dir"] / "evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        for section, content in report.items():
            f.write(f"\n{section}\n")
            f.write("-" * 50 + "\n")
            for key, value in content.items():
                f.write(f"{key}: {value}\n")


def main() -> Dict[str, Any]:
    """
    Main execution function for WGAN-GP training and evaluation.

    Returns:
        Dictionary containing results and evaluation metrics
    """
    # 1. Configuration Setup
    exploration_config = setup_config()
    print_config(exploration_config)

    # 2. Data Loading and Preprocessing
    print("\nStep 1: Data Loading and Preprocessing")
    print("-" * 50)
    data_info = load_and_process_data(exploration_config)

    # 3. Model Training and Evaluation
    print("\nStep 2: Model Training and Evaluation")
    print("-" * 50)
    results = train_and_evaluate_model(exploration_config)

    # 4. Store Dataset Information
    results["data_info"] = {
        "n_features": data_info.n_features,
        "timestamp": datetime.now().isoformat(),
        "original_samples": len(data_info.scaled_data),
        "control_samples": (data_info.scaled_data["target"] == 0).sum(),
        "covid_samples": (data_info.scaled_data["target"] == 1).sum(),
    }

    # 5. Generate Visualizations
    print("\nStep 3: Generating Visualizations")
    print("-" * 50)
    plot_evaluation_metrics(exploration_config, results)

    # 6. Generate Report
    print("\nStep 4: Generating Summary Report")
    print("-" * 50)
    generate_summary_report(exploration_config, results)

    # 7. Final Summary
    print("\nAnalysis Summary:")
    print("-" * 50)
    print(f"Results saved in: {exploration_config['results_dir']}")
    print(f"Total samples processed: {len(data_info.scaled_data)}")
    print(f"Features analyzed: {data_info.n_features}")
    print(f"Synthetic samples generated: {len(results['synthetic_data'])}")

    # Return results for interactive analysis
    return {
        "config": exploration_config,
        "data_info": data_info,
        "results": results,
        "evaluation_metrics": {
            "ks_stats": results["distribution_comparison"],
            "feature_matches": (
                results["distribution_comparison"]["KS_Statistic"] < 0.1
            ).sum(),
            "quality_score": (
                results["distribution_comparison"]["KS_Statistic"] < 0.1
            ).mean()
            * 100,
        },
    }


# Execute if running as script
if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Run main function
    analysis_results = main()

    # Print final quality score
    quality_score = analysis_results["evaluation_metrics"]["quality_score"]
    print(f"\nFinal Quality Score: {quality_score:.2f}%")
