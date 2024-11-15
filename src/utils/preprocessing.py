import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch
from torch.utils.data import TensorDataset
import numpy as np


def load_and_process_data(filepath):
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)
    return df

def scale_data(data):
    """
    Scale data using RobustScaler with improved handling of outliers
    """
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Additional outlier handling
    # scaled_data = np.clip(scaled_data, -5, 5)
    
    return scaled_data, scaler

### ORIGINAL DID NOT INCLUDE np.clip(scaled_data, -5, 5) ###
### np.clip provides a way to handle outliers by setting them to a specific value

# def scale_data(data):
#     # Initialize a RobustScaler
#     scaler = RobustScaler()
#     scaled_data = scaler.fit_transform(data)

#     return scaled_data, scaler

def process(filepath, split_ratio=0.2):
    # Load and process the data
    data = load_and_process_data(filepath)

    cutoff = len(data) * split_ratio
    cutoff = int(cutoff)

    leftout_df = data.copy().iloc[:cutoff] # Leave out for testing
    data = data.iloc[cutoff:]

    print()
    print(f"# of rows used for WGAN: {len(data)}")
    print(f"# of rows left out: {len(leftout_df)}")
    print()

    scaled_data, scaler = scale_data(data)

    tensor_data = torch.Tensor(scaled_data)
    dataset = TensorDataset(tensor_data)

    return dataset, tensor_data, data, leftout_df, scaler, scaled_data.shape[1]
