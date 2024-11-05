"""
WGAN-GP Training and Evaluation Script

This script demonstrates the training and evaluation of a WGAN-GP model for synthetic data generation.
The model's performance is evaluated against various metrics and visualized through statistical comparisons and t-SNE plots.
"""

import warnings

warnings.filterwarnings("ignore")

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import KFold
from scipy.stats import ks_2samp
from sklearn.manifold import TSNE
import config

# Local modules
from src.models.data_augmentation.GAN_v3 import train_and_generate
from src.utils.preprocessing_v3 import process
from src.utils.evaluation_v3 import (
    compare_distributions,
    compare_statistics,
    generate_tsne,
)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load and preprocess data
print("Loading and preprocessing data...")
scaled_data, scaler, _ = process(config.DATA_PATH)

# Train model and generate synthetic data
print("Training model and generating synthetic data...")
original_data, synthetic_data = train_and_generate()

# Statistical Comparison
print("\nPerforming statistical comparisons...")
stats_comparison = compare_statistics(original_data, synthetic_data)

# KS Test
print("\nPerforming KS Test...")
ks_results = compare_distributions(original_data, synthetic_data)

# Using the function and saving the plot
print("\nGenerating t-SNE visualization...")
generate_tsne(original_data, synthetic_data)
plt.savefig(config.RESULT_DIR / "tsne_visualization.png")
print("t-SNE visualization saved.")
# Save results
with open(config.RESULT_DIR / "results_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"Total original samples: {len(original_data)}\n")
    f.write(f"Total synthetic samples: {len(synthetic_data)}\n")
    f.write(
        f"Good KS features (%): {(ks_results['P-Value'] > 0.05).mean() * 100:.2f}\n"
    )

# Plotting KS results
plt.figure(figsize=(10, 6))
sns.histplot(ks_results["KS Statistic"], kde=True)
plt.title("KS Statistic Distribution: Original vs Synthetic Data")
plt.xlabel("KS Statistic")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig(config.RESULT_DIR / "ks_statistics_distribution.png")
plt.show()

# Generate and save feature distributions
for column in original_data.columns:
    if column not in ["data_type", "fold", "sample_id"]:
        plt.figure()
        sns.kdeplot(original_data[column], label="Original")
        sns.kdeplot(synthetic_data[column], label="Synthetic")
        plt.legend()
        plt.title(f"Distribution for {column}")
        plt.xlabel(column)
        plt.ylabel("Density")
        plt.grid(True)
        plt.savefig(config.RESULT_DIR / f"feature_distribution_{column}.png")
        plt.close()

print("Analysis complete. Results saved in:", config.RESULT_DIR)
