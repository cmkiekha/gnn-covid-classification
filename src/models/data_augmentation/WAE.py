import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tqdm import tqdm

from src.utils.preprocessing import *

class WAE(nn.Module):
    def __init__(self, original_dim, intermediate_dim=512, latent_dim=2):
        super().__init__()
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(original_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, latent_dim) # WAE does not split into mu and log(var)
        )
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, original_dim)
        )

    def forward(self, x):
        # Encode the input data
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z

def train(wae, data_loader, epochs=250, lambda_pen=10):
    # Initialize the optimizer (Adam)
    optimizer = optim.Adam(wae.parameters())
    
    # Set the WAE to training mode
    wae.train()
    
    for _ in tqdm(range(epochs), desc="Training WAE"):
        for batch_features, in data_loader:
            # Clear the gradients
            optimizer.zero_grad()
            
            # Forward pass through the WAE
            reconstruction, z = wae(batch_features[0])
            
            # Compute the reconstruction loss
            reconstruction_loss = nn.MSELoss()(reconstruction, batch_features[0])
            z_prior = torch.randn_like(z)
            
            wasserstein_distance = torch.mean(torch.abs(z - z_prior))
            loss = reconstruction_loss + lambda_pen * wasserstein_distance
            
            # Backpropagate the gradients
            loss.backward()
            
            # Update the model parameters
            optimizer.step()

def generate_new_data(wae, num_synthetics=100):
    wae.eval()
    
    # Create num_synthetics samples from dist
    z = torch.randn(num_synthetics, 2)

    # Decode the samples from the latent space
    with torch.no_grad():
        reconstruction = wae.decoder(z)
    
    return reconstruction.numpy()

def train_wae(dataset, original_dim, batch_size=32, epochs=100):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    wae = WAE(original_dim)
    train(wae, dataloader, epochs)
    return wae

def generate_wae(vae, columns, scaler, num_synthetics=100):
    new_data = generate_new_data(vae, num_synthetics)
    new_data_unscaled = scaler.inverse_transform(new_data)
    new_df = pd.DataFrame(new_data_unscaled, columns=columns)
    return new_df
