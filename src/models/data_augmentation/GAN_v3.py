import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path

# Import configuration settings
from src.utils.preprocessing_v3 import process
from src.utils.evaluation_v3 import recenter_data

import config

print(config.BATCH_SIZE)


class Generator(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(Generator, self).__init__()
        self.input_dim = input_dim  # Store input_dim as an attribute
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Critic(nn.Module):
    def __init__(self, input_dim: int):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_gradient_penalty(critic, real_data, fake_data, device):
    alpha = torch.rand(real_data.size(0), 1, device=device, requires_grad=True)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated_scores = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=interpolated_scores,
        inputs=interpolated,
        grad_outputs=torch.ones(interpolated_scores.size(), device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_wgan_gp(
    data_loader,
    generator,
    critic,
    g_optimizer,
    c_optimizer,
    device,
    epochs,
    lambda_gp=10,
):
    generator.train()
    critic.train()
    g_losses, c_losses = [], []

    for epoch in range(epochs):
        for data in data_loader:
            real_samples = data.to(device)
            fake_samples = generator(
                torch.randn(real_samples.size(0), generator.input_dim).to(device)
            )

            # ----- Critic update -----
            critic.zero_grad()
            real_scores = critic(real_samples)
            fake_scores = critic(fake_samples.detach())
            gp = compute_gradient_penalty(critic, real_samples, fake_samples, device)
            c_loss = fake_scores.mean() - real_scores.mean() + lambda_gp * gp
            c_loss.backward()  # Update the critic
            c_optimizer.step()
            c_losses.append(c_loss.item())

            # ----- Generator update -----
            generator.zero_grad()
            # Recompute fake samples for the generator loss
            fake_samples = generator(
                torch.randn(real_samples.size(0), generator.input_dim).to(device)
            )
            fake_scores = critic(fake_samples)
            g_loss = -fake_scores.mean()
            g_loss.backward()  # Update the generator
            g_optimizer.step()
            g_losses.append(g_loss.item())

        # Optional: Logging every 10 epochs
        if epoch % 10 == 0 or epoch == epochs - 1:
            plot_losses(g_losses, c_losses, f"Epoch {epoch}")

    return g_losses, c_losses


def plot_losses(g_losses, c_losses, epoch):
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label="Generator Loss", color="blue")
    plt.plot(c_losses, label="Critic Loss", color="red")
    plt.title(f"Training Losses - {epoch}")
    plt.xlabel("Training Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{config.RESULT_DIR}/loss_plot_{epoch}.png")
    plt.close()


def train_and_generate():
    print("Using the following settings:")
    print(f"File path: {config.DATA_PATH}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.DEBUG_EPOCHS}")
    print(f"Device: {config.DEVICE}")
    print(f"Number of splits for K-Fold validation: {config.CV_N_SPLITS}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print("Initializing training and data generation process.")

    # Get processed data and ensure `scaled_data` is a DataFrame
    scaled_data, scaler, original_data = process(config.DATA_PATH)

    # Ensure `scaled_data` is a DataFrame with correct column names
    if isinstance(scaled_data, np.ndarray):
        scaled_data = pd.DataFrame(scaled_data, columns=original_data.columns)

    kfold = KFold(
        n_splits=config.CV_N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE
    )
    synthetic_samples = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(scaled_data)):
        train_data = torch.tensor(
            scaled_data.iloc[train_idx].values, dtype=torch.float32
        ).to(config.DEVICE)

        train_loader = DataLoader(
            train_data, batch_size=config.BATCH_SIZE, shuffle=True
        )
        generator = Generator(original_data.shape[1], original_data.shape[1]).to(
            config.DEVICE
        )
        critic = Critic(original_data.shape[1]).to(config.DEVICE)
        g_optimizer = Adam(generator.parameters(), lr=config.LEARNING_RATE)
        c_optimizer = Adam(critic.parameters(), lr=config.LEARNING_RATE)

        g_losses, c_losses = train_wgan_gp(
            train_loader,
            generator,
            critic,
            g_optimizer,
            c_optimizer,
            config.DEVICE,
            config.DEBUG_EPOCHS,
        )

        noise = torch.randn(len(val_idx), original_data.shape[1], device=config.DEVICE)
        synthetic_data = generator(noise).detach().cpu().numpy()

        # Create synthetic_df using the same columns as scaled_data
        synthetic_data = recenter_data(
            synthetic_data, scaled_data.iloc[val_idx].values, scaler
        )
        synthetic_df = pd.DataFrame(synthetic_data, columns=scaled_data.columns)
        synthetic_df["data_type"] = "synthetic"
        synthetic_df["fold"] = fold
        synthetic_samples.append(synthetic_df)

    combined_data = pd.concat([original_data] + synthetic_samples, ignore_index=True)

    if config.SAVE_INFO:
        results_path = Path(config.RESULT_DIR) / "augmented_data.csv"
        combined_data.to_csv(results_path, index=False)
        print(f"Augmented data saved to {results_path}")

    return combined_data, original_data
