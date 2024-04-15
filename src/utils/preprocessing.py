import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset

def load_and_process_data(filepath):
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)
    return df

def scale_data(data):
    # Initialize a StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    return scaled_data, scaler

def process(filepath):
    # Load and process the data
    data = load_and_process_data(filepath)
    scaled_data, scaler = scale_data(data)

    tensor_data = torch.Tensor(scaled_data)
    dataset = TensorDataset(tensor_data)

    return dataset, tensor_data, data, scaler, scaled_data.shape[1]
