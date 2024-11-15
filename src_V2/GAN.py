"""
WGAN-GP implementation for protein/transcript data augmentation.

This module implements a Wasserstein GAN with Gradient Penalty for generating
synthetic protein/transcript data. It includes the Generator and Critic networks,
training logic, and K-fold cross-validation implementation.

Key Components:
    - Generator: Network that generates synthetic samples
    - Critic: Network that evaluates sample authenticity
    - train_wgan_gp: Main training loop with gradient penalty
    - train_and_generate: High-level function combining training and generation with K-fold CV

"""

# GAN.py
import sys
import os

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

import pandas as pd
from tqdm import tqdm
from datetime import datetime
import json 
from pathlib import Path
from typing import Tuple, List
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

from src_v2.utils import load_raw_data
from src_v2.models import Generator, Critic, compute_gradient_penalty
from src_v2.config import (
    DEVICE,
    CV_N_SPLITS,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    RESULT_DIR,
)
from src_v2.utils import load_raw_data
from src_v2.eval import evaluate_gan_samples
from src.utils.evaluation import recenter_data


def train_wgan_gp(
    data_loader: DataLoader,
    generator: Generator,
    critic: Critic,
    g_optimizer: Adam,
    c_optimizer: Adam,
    device: torch.device,
    epochs: int = 10,
    critic_iterations: int = 5,
    lambda_gp: float = 10,
    noise_multiplier: float = 1.0,
) -> Tuple[List[float], List[float]]:
    """
    Train WGAN-GP with enhanced diversity and stability measures.

    Combines best practices from both implementations:
    1. Fresh noise for each critic iteration (diversity)
    2. Spectral normalization in critic (stability)
    3. Dynamic noise scaling (exploration)
    4. Gradient norm monitoring

    Args:
        data_loader: DataLoader for training data
        generator: Generator network
        critic: Critic network
        g_optimizer: Generator optimizer
        c_optimizer: Critic optimizer
        device: Computation device
        epochs: Number of training epochs
        critic_iterations: Number of critic updates per generator update
        lambda_gp: Gradient penalty coefficient
        noise_multiplier: Scale factor for input noise

    Returns:
        Tuple of (generator_losses, critic_losses)
    """

    generator.train()
    critic.train()

    g_losses = []
    c_losses = []

    for epoch in tqdm(range(epochs), desc="Training WGAN-GP"):
        epoch_g_loss = 0
        epoch_c_loss = 0
        n_batches = 0

        for i, data in enumerate(data_loader):
            real_samples = data[0].to(device)
            batch_size = real_samples.size(0)

            # Train Critic
            for _ in range(critic_iterations):
                c_optimizer.zero_grad()

                # Generate diverse fake samples with scaled noise
                noise = torch.randn(batch_size, generator.input_dim, device=device)
                noise = noise * noise_multiplier
                fake_samples = generator(noise)

                # Compute critic scores
                real_validity = critic(real_samples)
                fake_validity = critic(fake_samples.detach())

                # Gradient penalty
                gp = compute_gradient_penalty(
                    critic, real_samples, fake_samples, device
                )

                # Critic loss with Wasserstein distance
                c_loss = fake_validity.mean() - real_validity.mean() + lambda_gp * gp

                c_loss.backward()
                c_optimizer.step()
                epoch_c_loss += c_loss.item()

            # Train Generator
            g_optimizer.zero_grad()

            # Generate new samples with fresh noise
            noise = (
                torch.randn(batch_size, generator.input_dim, device=device)
                * noise_multiplier
            )
            fake_samples = generator(noise)
            fake_validity = critic(fake_samples)

            # Generator loss
            g_loss = -fake_validity.mean()

            g_loss.backward()
            g_optimizer.step()

            # Store losses
            g_losses.append(g_loss.item())
            c_losses.append(c_loss.item())
            epoch_g_loss += g_loss.item()
            n_batches += 1

            # Progress reporting
            if i % 50 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}] Batch [{i}] "
                    f"G_loss: {g_loss.item():.4f} C_loss: {c_loss.item():.4f}"
                )

        # Epoch summary
        print(f"\nEpoch {epoch} Summary:")
        if n_batches == 0:
            print("WARNING: 0 Batches. Unable to compute loss.")
        else:
            print(f"Average G_loss: {epoch_g_loss/n_batches:.4f}")
            print(f"Average C_loss: {epoch_c_loss/n_batches:.4f}")

    # Visualize training losses
    plt.figure(figsize=(12, 5))

    # Plot Generator Loss
    plt.subplot(1, 2, 1)
    plt.plot(g_losses, label="Generator Loss", color="blue", alpha=0.7)
    plt.title("Generator Loss Over Training")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # Plot Critic Loss
    plt.subplot(1, 2, 2)
    plt.plot(c_losses, label="Critic Loss", color="red", alpha=0.7)
    plt.title("Critic Loss Over Training")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot combined losses
    plt.figure(figsize=(10, 6))
    plt.plot(g_losses, label="Generator Loss", color="blue", alpha=0.7)
    plt.plot(c_losses, label="Critic Loss", color="red", alpha=0.7)
    plt.title("Training Losses")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    return g_losses, c_losses


def generate(generator, n_samples, n_features, scaler, device):
    latent_samples = torch.randn(num_samples, n_features, device=device)
    with torch.no_grad():
        generator.eval()
        synthetic_samples = generator(latent_samples).cpu().numpy()
        generator.train()

    # Return generated data back to original scale
    synthetic_unscaled = scaler.inverse_transform(synthetic_samples)
    return synthetic_unscaled


def train_and_generate(
    filepath: str,
    batch_size: int = 32,
    epochs: int = 20,
    device: str = "cpu",
    n_splits: int = 3,
    learning_rate: float = 0.001,
    save_info: bool = True,
    save_path=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train WGAN-GP and generate synthetic data using K-Fold validation.

    This function implements:
    1. K-fold cross-validation for robust training
    2. Proper data labeling and tracking
    3. Synthetic data generation with recentering
    4. Detailed progress reporting and data summaries

    Args:
        filepath: Path to the dataset
        batch_size: Training batch size
        epochs: Number of training epochs
        device: Computing device ('cpu' or 'cuda')
        n_splits: Number of folds for K-Fold validation
        learning_rate: Learning rate for optimizers
        save_info: Whether to save data info to JSON

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (complete_dataset, original_data)
            - complete_dataset: Combined original and synthetic data
            - original_data: Original data only, for testing
    """
    raw_data = load_raw_data(filepath, to_tensor=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Initialize K-fold
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Training and generation for each fold
    for fold, (train_idx, val_idx) in enumerate(kfold.split(raw_data), 1):
        print(f"\nProcessing fold {fold}/{n_splits}")
        print(f"Training samples: {len(train_idx)}")
        print(f"Validation samples: {len(val_idx)}")

        # Split raw data into tr/te
        tr_data = raw_data[train_idx]
        te_data = raw_data[te_idx]

        # tr_controls = tr_data[y == 0]

        # Now do scaling
        scaler = RobustScaler()
        tr_data_scaled = scaler.fit_transform(tr_data)

        ##### Now the GAN stuff starts
        train_loader = DataLoader(
            tr_data_scaled, batch_size=batch_size, shuffle=True, drop_last=False
        )
        print(f"Train Loader Len: {len(train_loader)}")

        # Initialize models
        generator = Generator(n_features, n_features).to(device)
        critic = Critic(n_features).to(device)

        # Initialize optimizers with beta parameters for stability
        g_optimizer = Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
        c_optimizer = Adam(critic.parameters(), lr=learning_rate, betas=(0.5, 0.9))

        # Train the models
        g_losses, c_losses = train_wgan_gp(
            train_loader,
            generator,
            critic,
            g_optimizer,
            c_optimizer,
            device,
            epochs=epochs,
        )

        ### After GAN training

        # Generate synthetic samples
        num_samples = len(val_idx)
        synthetic_samples = generate(generator, num_samples, n_features, scaler, device)
        synthetic_samples = recenter_data(synthetic_samples, tr_data)

        #### Saving our splits into a format like this:

        # save_path/fold1/tr_data.pt
        # save_path/fold1/te_data.pt
        # save_path/fold1/synthetic_data.pt

        # save_path/fold2/tr_data.pt
        # save_path/fold2/te_data.pt
        # save_path/fold2/synthetic_data.pt

        # NOTE: this data is on the original scale; will need to be re-scaled if you e.g. put it into a classifier
        this_save_path = save_path / f"fold{fold}"
        os.mkdir(
            this_save_path, parents=True, exist_ok=True
        )  # Makes the folder if it doesn't exist
        torch.save(tr_data, this_save_path / "tr_data.pt")
        torch.save(te_data, this_save_path / "te_data.pt")
        torch.save(synthetic_samples, this_save_path / "synthetic_data.pt")

        # Plot losses for this fold
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(g_losses, label=f"Generator Loss - Fold {fold}")
        plt.title("Generator Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(c_losses, label=f"Critic Loss - Fold {fold}")
        plt.title("Critic Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print(f"Completed fold {fold}. Generated {len(fold_df)} samples.")

    # Combine all data
    synthetic_df = pd.concat(generated_samples, ignore_index=True)
    final_df = pd.concat([original_data, synthetic_df], ignore_index=True)

    # Save data information if requested
    if save_info:
        data_info = {
            "n_original": len(original_data),
            "n_synthetic": len(synthetic_df),
            "n_folds": n_splits,
            "feature_names": [
                col
                for col in scaled_data.columns
                if col not in ["data_type", "fold", "sample_id"]
            ],
            "synthetic_fold_sizes": synthetic_df.groupby("fold").size().to_dict(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        if isinstance(filepath, str):
            info_path = Path(filepath).parent / "data_info.json"
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(data_info, f, indent=2)

    print("\nGeneration Summary:")
    print(f"Original samples: {len(original_data)}")
    print(f"Synthetic samples: {len(synthetic_df)}")
    print(f"Total samples: {len(final_df)}")
    print("\nSynthetic samples per fold:")
    print(synthetic_df.groupby("fold").size())

    return final_df, original_data


if __name__ == "__main__":
    # Usage of the function assuming preprocessing_v3.process() returns the required train_df
    # For example, let's assume the process function is called as follows:
    filepath = "/Users/carolkiekhaefer10-2023/Documents/GitHub/gnn-covid-classification/data/data_combined_controls.csv"

    # Read and inspect the data first
    print("\n=== Data Inspection ===")
    df = load_raw_data(filepath)
    print("\nColumns in dataset:")
    print(df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
    print("\nShape of dataset:", df.shape)

    train_and_generate(
        filepath,
        BATCH_SIZE,
        EPOCHS,
        DEVICE,
        CV_N_SPLITS,
        LEARNING_RATE,
        save_path=RESULT_DIR,
    )

    # evaluate_wgan_samples(RESULT_DIR, num_folds=CV_N_SPLITS)


## If we just run `python GAN.py` --> will produce your folders fold1/.... fold2/... and loss plots, tsne, ks
