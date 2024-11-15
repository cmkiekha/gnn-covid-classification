"""
This file contains the configuration for the project.
Central use keeps the script general and adaptable to changes in data paths or settings.
"""

# config.py
import torch
from pathlib import Path

# Paths
DATA_PATH = Path("/Users/carolkiekhaefer10-2023/Documents/GitHub/gnn-covid-classification/data/data_combined_controls.csv")
RESULT_DIR = Path("results")

# Model Configurations
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0002
RANDOM_STATE = 42
CV_N_SPLITS = 3
TEST_SIZE = 0.2

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Save Results Flag
SAVE_INFO = True

