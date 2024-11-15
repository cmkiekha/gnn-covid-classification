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
BATCH_SIZE = 16
DEV_EPOCHS = 300
DEBUG_EPOCHS = 3
RANDOM_STATE = 42
CV_N_SPLITS = 3
LEARNING_RATE = 0.0002
TEST_SIZE = 0.2
SAMPLE_COUNT_TO_GENERATE = 100

GENERATOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 2e-4
ADAM_BETA1 = 0.0
ADAM_BETA2 = 0.9

# Device config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Saving config
SAVE_INFO = True  # Set this to False if you do not wish to automatically save results
