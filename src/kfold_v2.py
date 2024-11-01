"""
Data leakage:
Loop over splits:
    Make tr/te of data corresponding to split

    Train WGAN only on train data
    Generated samples from WGAN to make synthetic data
Aggregate samples

Tr/te split of augmented data
Train classifier on tr
Eval on test

No leakage:
Loop over splits:
    Make tr/te of data corresponding to split

    Train WGAN only on train data
    Generated samples from WGAN to make synthetic data

    Train classifier on tr split + synthetic data

    Evaluate classifier on te split
Average over test splits

How Leakage is AvoidedLeakage is avoided in the “Approach without 
Data Leakage” by ensuring that the synthetic data generatedby the 
WGAN is used strictly within the same fold where it was generated. 
Each fold’s training and testingdata are completely isolated from 
one another, meaning that no information from the test sets can 
influencethe training process, including the generation and utilization 
of synthetic data. This method ensures thatthe integrity and independence 
of the test sets are maintained, leading to more reliable and generalizable 
evaluation metrics.This approach respects the fundamental principle of 
cross-validation where each test set must remain entirelyunseen by the model 
during training, thereby preventing any form of leakage that could skew the results.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Optional, Dict, Any
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import xgboost as xgb

from src.utils.preprocessing import process
from src.utils.evaluation import recenter_data



## DIAGRAM COMPARISONS OF DATA LEAKAGE VS NO DATA LEAKAGE



def load_raw_data(dataset_path):
    # all this fxn should do is directly read the CSV (with all data, having x and y variables)

    if not len(dataset_path):
        # dataset_path = "/Users/carolkiekhaefer10-2023/Documents/COVID-19_CKOA/4-13-24/data/data_combined_controls.csv"
        dataset_path = "data_combined.csv"  # Whatever file has all of the data, including the labels
    print(f"\nLoading dataset from path: {dataset_path}")
    # Validate file path
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    # Load and validate data
    raw_data = pd.read_csv(filepath)
    # Remove any non-numeric columns
    numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in dataset")

    data = raw_data[numeric_cols]

    data_y = data.pop("Group")
    return data, data_y  # NOTE: may need to do some further preprocessing...


def do_kfold_with_augmentation(
    dataset_path, batch_size=32, epochs=100, learning_rate=1e-3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_data_x, raw_data_y = load_raw_data(dataset_path)

    all_f1 = []

    kfold = KFold(
        n_splits=n_splits, shuffle=True, random_state=42
    )  # might want to stratify here
    for fold, (train_idx, val_idx) in enumerate(kfold.split(scaled_data), 1):
        print(f"\nProcessing fold {fold}/{n_splits}")
        print(f"Training samples: {len(train_idx)}")
        print(f"Validation samples: {len(val_idx)}")

        # Actually split the data
        train_x = raw_data_x[train_idx]
        train_y = raw_data_y[train_idx]

        test_x = raw_data_x[val_idx]
        test_y = raw_data_y[val_idx]

        # Scale data
        scaler = RobustScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)

        # Turn into datasets/dataloaders
        train_dataset = torch.utils.data.TensorDataset(train_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        print(f"Train Loader Len: {len(train_loader)}")
        train_dataset = torch.utils.data.TensorDataset(test_tensor)
        train_loader = DataLoader(
            test_tensor, batch_size=batch_size, shuffle=True, drop_last=False
        )
        print(f"Train Loader Len: {len(train_loader)}")

        # Only get the control data for the purposes of training the WGAN
        train_x_control = train_x[train_y == 0]
        train_x_control_dataset = torch.utils.data.TensorDataset(train_x_control)
        train_x_control_loader = DataLoader(
            train_x_control_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        # Now actually train WGAN on the train data, followed by generating samples
        # note: synthetic samples all have y=0
        _, synthetic_samples = train_and_generate_v2(
            train_x_control_loader, scaler, epochs=epochs, learning_rate=learning_rate
        )

        # Create augmented train set (x-values)
        augmented_train_x = torch.vstack(
            (train_x, synthetic_samples)
        )  # this should just stack together the train and synthetic data (i.e., stack by rows)

        # adds labels to the training data (all y=0)
        augmented_train_y = torch.cat(
            (train_y, torch.zeros(shape(synthetic_samples[0]), dtype=int))
        )

        # Train clasifier on augmented data
        xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
        xgb_model.fit(augmented_train_x, augmented_train_y)

        # Now finally do evaluation
        y_test_pred = xgb_model.pred(test_x)
        f1 = f1_score(test_y, y_test_pred)
        all_f1.append(f1)


print("Finished running k fold :) ")
print(f"Average F1: {np.mean(all_f1)}")
