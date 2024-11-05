"""
This file contains the configuration for the project.
Central use keeps the script general and adaptable to changes in data paths or settings.
"""

import torch
from pathlib import Path

# Path config
DATA_PATH = "/Users/carolkiekhaefer10-2023/Documents/GitHub/gnn-covid-classification/data/data_combined_controls.csv"
RESULT_DIR = Path("results")  # Using Pathlib for path operations

# Model config
BATCH_SIZE = 32
DEV_EPOCHS = 100
DEBUG_EPOCHS = 5
RANDOM_STATE = 42
CV_N_SPLITS = 3
LEARNING_RATE = 0.0002

# Device config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Saving config
SAVE_INFO = True  # Set this to False if you do not wish to automatically save results

