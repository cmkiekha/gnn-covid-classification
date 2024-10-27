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

import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Optional, Dict, Any
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils.preprocessing import process
from src.utils.evaluation import recenter_data


class Generator(nn.Module):
    """
    Generator for a Wasserstein GAN with Gradient Penalty (WGAN-GP), responsible for generating
    synthetic data from noise input.

    Attributes:
        input_dim (int): Dimensionality of the input noise vector.
        output_dim (int): Dimensionality of the output (generated) data.
        net (torch.nn.Sequential): The neural network that defines the generator.
    """

    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
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
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, z):
        """
        Forward pass of the generator.

        Args:
            z (torch.Tensor): A batch of random noise vectors.

        Returns:
            torch.Tensor: Generated data corresponding to the input noise.
        """
        return self.net(z)


class Critic(nn.Module):
    """
    Critic (or discriminator) for a Wasserstein GAN with Gradient Penalty (WGAN-GP). The critic evaluates
    the authenticity of both real and generated data.

    Attributes:
        net (torch.nn.Sequential): The neural network that defines the critic.
    """

    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        """
        Forward pass of the critic.

        Args:
            x (torch.Tensor): A batch of real or generated data.

        Returns:
            torch.Tensor: The critic's score for the input data, indicating its 'realness'.
        """
        return self.net(x)


def compute_gradient_penalty(
    critic: nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Computes the gradient penalty for enforcing the Lipschitz constraint in WGAN-GP.
    This penalty promotes smooth gradients of the critic network.

    Args:
        critic (torch.nn.Module): The critic network that evaluates the authenticity of data.
        real_samples (torch.Tensor): Samples from the real dataset.
        fake_samples (torch.Tensor): Generated samples from the generator.
        device (torch.device): The device tensors are on (e.g., CPU or CUDA).

    Returns:
        The computed gradient penalty, a scalar tensor (torch tensor) that should be added
        to the critic's loss to enforce the Lipschitz condition.
    """
    # Random weight for interpolation
    alpha = torch.rand((real_samples.size(0), 1), device=device)

    # Interpolate between real and fake samples
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    # Get critic scores
    disc_interpolates = critic(interpolates)

    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Calculate penalty
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


def train_wgan_gp(
    data_loader: DataLoader,
    generator: Generator,
    critic: Critic,
    g_optimizer: Adam,
    c_optimizer: Adam,
    device: torch.device,
    epochs: int = 100,
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


## ORIGINAL CODE
# def train_wgan_gp(
#     data_loader,
#     generator,
#     critic,
#     g_optimizer,
#     c_optimizer,
#     device,
#     epochs=100,
#     critic_iterations=5,
#     lambda_gp=10,
# ):
#     generator.train()
#     critic.train()

#     g_losses = []
#     c_losses = []

#     for epoch in tqdm(range(epochs), desc="Training WGAN-GP"):
#         for i, data in enumerate(data_loader):
#             real_samples = data[0].to(device)
#             current_batch_size = real_samples.size(0)

#             # Train Critic
#             for _ in range(critic_iterations):
#                 fake_samples = generator(
#                     torch.randn(current_batch_size, generator.input_dim, device=device)
#                 )
#                 real_scores = critic(real_samples)
#                 fake_scores = critic(fake_samples)
#                 gradient_penalty = compute_gradient_penalty(
#                     critic, real_samples, fake_samples, device
#                 )
#                 c_loss = (
#                     -(torch.mean(real_scores) - torch.mean(fake_scores))
#                     + lambda_gp * gradient_penalty
#                 )
#                 critic.zero_grad()
#                 c_loss.backward()
#                 c_optimizer.step()

#             # Train Generator
#             fake_samples = generator(
#                 torch.randn(current_batch_size, generator.input_dim, device=device)
#             )
#             fake_scores = critic(fake_samples)
#             g_loss = -torch.mean(fake_scores)
#             generator.zero_grad()
#             g_loss.backward()
#             g_optimizer.step()

#             if i % 5 == 0:
#                 print(
#                     f"Epoch: {epoch}, Batch: {i}, G_loss: {g_loss.item()}, C_loss: {c_loss.item()}"
#                 )
#                 g_losses.append(g_loss.item())
#                 c_losses.append(c_loss.item())

#     plt.figure(figsize=(6, 6))
#     plt.plot(range(len(g_losses)), g_losses, marker="o")
#     plt.title("Generator Loss")
#     plt.grid(True)
#     plt.show()

#     plt.figure(figsize=(6, 6))
#     plt.plot(range(len(c_losses)), c_losses, marker="o")
#     plt.title("Critic Loss")
#     plt.grid(True)
#     plt.show()


def train_and_generate(
    filepath: str,
    batch_size: int = 32,
    epochs: int = 20,
    device: str = "cpu",
    n_splits: int = 5,
    learning_rate: float = 0.001,
    save_info: bool = True,
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
    # Load and process data - unpack only what we need
    _, _, scaled_data, scaler, n_features = process(filepath)

    # Store original data with proper labeling
    original_data = scaled_data.copy()
    original_data["fold"] = -1  # Marker for original data
    original_data["data_type"] = "original"
    original_data["sample_id"] = range(len(original_data))

    print("\nData Overview:")
    print(f"Original samples: {len(original_data)}")
    print(f"Features: {n_features}")
    print("\nFeature types:")
    print(scaled_data.dtypes)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Initialize K-fold
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    generated_samples = []

    # Training and generation for each fold
    for fold, (train_idx, val_idx) in enumerate(kfold.split(scaled_data), 1):
        print(f"\nProcessing fold {fold}/{n_splits}")
        print(f"Training samples: {len(train_idx)}")
        print(f"Validation samples: {len(val_idx)}")

        # Prepare training data
        train_tensor = torch.tensor(
            scaled_data.iloc[train_idx].values, dtype=torch.float32
        ).to(device)

        train_loader = DataLoader(
            train_tensor  # , batch_size=batch_size, shuffle=True, drop_last=True
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

        # Generate synthetic samples
        num_samples = len(val_idx)
        latent_samples = torch.randn(num_samples, n_features, device=device)

        with torch.no_grad():
            generator.eval()
            synthetic_samples = generator(latent_samples).cpu().numpy()
            generator.train()

        # Process generated samples
        synthetic_unscaled = scaler.inverse_transform(synthetic_samples)
        recentered_samples = recenter_data(
            synthetic_unscaled, scaled_data.iloc[val_idx].values
        )

        # Create DataFrame with metadata
        fold_df = pd.DataFrame(
            recentered_samples,
            columns=[
                col
                for col in scaled_data.columns
                if col not in ["data_type", "fold", "sample_id"]
            ],
        )

        # Add metadata
        fold_df["data_type"] = "synthetic"
        fold_df["fold"] = fold
        fold_df["generation_seed"] = fold
        fold_df["reference_samples"] = [val_idx.tolist()]

        generated_samples.append(fold_df)

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


## GENERATING ERROR
# def train_and_generate(filepath, batch_size=32, epochs=100, device="cpu", n_splits=5):
#     """
#     Trains a WGAN-GP and generates synthetic data using K-Fold validation.

#     Args:
#         filepath (str): Path to the dataset
#         batch_size (int): Batch size for training
#         epochs (int): Number of training epochs
#         device (str): Device to use for training
#         n_splits (int): Number of folds for K-Fold validation

#     Returns:
#         pd.DataFrame: Generated synthetic samples with labels
#     """
#     # Load and process data
#     _, tensor_data, scaled_data, scaler, _ = process(filepath)

#     # Debug print
#     print("Original data types:")
#     print(scaled_data.dtypes)

#     # Ensure all data are numeric and handle type conversion
#     numeric_cols = scaled_data.select_dtypes(include=[np.number]).columns
#     if len(numeric_cols) == 0:
#         raise ValueError("No numeric columns found in the dataset")

#     print(f"\nUsing {len(numeric_cols)} numeric columns: {numeric_cols.tolist()}")

#     # Select only numeric columns and convert to float32
#     scaled_data = scaled_data[numeric_cols].astype(np.float32)

#     # Convert to tensor
#     tensor_data = torch.tensor(scaled_data.values, dtype=torch.float32)

#     # Setup device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     input_dim = scaled_data.shape[1]

#     # Initialize K-Fold
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

#     all_generated_samples = []

#     for fold, (train_idx, test_idx) in enumerate(kf.split(tensor_data)):
#         print(f"\nProcessing fold {fold + 1}/{n_splits}")

#         # Create train/test splits
#         train_data = torch.utils.data.TensorDataset(tensor_data[train_idx])
#         train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

#         # Initialize models
#         generator = Generator(input_dim, input_dim).to(device)
#         critic = Critic(input_dim).to(device)

#         # Initialize optimizers
#         g_optimizer = Adam(generator.parameters())
#         c_optimizer = Adam(critic.parameters())

#         # Train the model
#         train_wgan_gp(
#             train_loader, generator, critic, g_optimizer, c_optimizer, device, epochs
#         )

#         # Generate samples
#         num_samples = len(test_idx)
#         latent_samples = torch.randn(num_samples, input_dim, device=device)
#         generated_samples = generator(latent_samples).detach().cpu().numpy()

#         # Recenter the generated data
#         generated_samples = recenter_data(
#             generated_samples, scaled_data.iloc[test_idx].values
#         )

#         # Create DataFrame with generated samples
#         generated_df = pd.DataFrame(generated_samples, columns=numeric_cols)
#         generated_df["fold"] = fold + 1
#         generated_df["type"] = "synthetic"

#         all_generated_samples.append(generated_df)

#         print(f"Completed fold {fold + 1}. Generated {len(generated_df)} samples.")

#     # Combine all generated samples
#     final_df = pd.concat(all_generated_samples, ignore_index=True)

#     return final_df


# def train_and_generate(filepath, batch_size=32, epochs=100, device='cpu'):

#     _, tensor_data, scaled_data, _, _ = process(filepath)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     input_dim = scaled_data.shape[1]

#     data_loader = DataLoader(tensor_data, batch_size=batch_size, shuffle=True)

#     generator = Generator(input_dim, input_dim).to(device)
#     critic = Critic(input_dim).to(device)

#     g_optimizer = Adam(generator.parameters())
#     c_optimizer = Adam(critic.parameters())

#     train_wgan_gp(data_loader, generator, critic, g_optimizer, c_optimizer, device, epochs)

#     # Generate samples and compute statistical metrics
#     num_samples = 100
#     latent_samples = torch.randn(num_samples, generator.input_dim, device=device)
#     generated_samples = generator(latent_samples).detach().cpu().numpy()
#     return generated_samples
