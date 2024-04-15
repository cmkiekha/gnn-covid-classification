import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tqdm import tqdm


class VAE(nn.Module):
    def __init__(self, original_dim, intermediate_dim=512, latent_dim=2):
        super().__init__()
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(original_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, latent_dim * 2)
        )
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, original_dim)
        )

    def reparameterize(self, mu, log_var):
        # Reparameterization trick to sample from the latent space
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        
        # Split the encoded representation into mean and log variance
        mu, log_var = encoded.chunk(2, dim=-1)
        
        # Sample from the latent space using reparameterization trick
        z = self.reparameterize(mu, log_var)
        
        # Decode the sampled latent representation
        return self.decoder(z), mu, log_var

def train(vae, data_loader, epochs=100, lr=1e-3):
    # Initialize the optimizer (Adam)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    vae.train()
    
    for _ in tqdm(range(epochs), desc="Training VAE"):
        for batch_features, in data_loader:
            # Clear the gradients
            optimizer.zero_grad()
            
            # Forward pass through the VAE
            reconstruction, mu, log_var = vae(batch_features[0])
            
            # Compute the reconstruction loss
            reconstruction_loss = nn.MSELoss()(reconstruction, batch_features[0])
            
            # Compute the KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = reconstruction_loss + kl_loss
            
            # Backpropagate the gradients
            loss.backward()
            
            # Update the model parameters
            optimizer.step()

def generate_new_data(vae, num_synthetics=100):
    vae.eval()
    
    # Create num_synthetics samples from dist
    z = torch.randn(num_synthetics, 2)

    # Decode the samples from the latent space
    with torch.no_grad():
        reconstruction = vae.decoder(z)
    
    return reconstruction.numpy()

def train_vae(dataset, original_dim, batch_size=32, epochs=100):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    vae = VAE(original_dim)
    train(vae, dataloader, epochs)
    return vae

def generate_vae(vae, columns, scaler, num_synthetics=100):
    new_data = generate_new_data(vae, num_synthetics)
    new_data_unscaled = scaler.inverse_transform(new_data)
    new_df = pd.DataFrame(new_data_unscaled, columns=columns)
    return new_df
