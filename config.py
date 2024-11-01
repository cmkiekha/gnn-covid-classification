import torch

# Path config
DATA_PATH = "data/data_combined_controls.csv"
RESULT_DIR = "results"

# Model config
BATCH_SIZE = 32
DEV_EPOCHS = 100
DEBUG_EPOCHS = 5
RANDOM_STATE = 42
CV_N_SPLITS = 3

# Device config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
