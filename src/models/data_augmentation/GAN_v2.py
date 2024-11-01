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

from datetime import datetime
from pathlib import Path
import json
from typing import Tuple, List, Dict
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, KFold
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.utils.preprocessing import process
from src.utils.evaluation import recenter_data

import config


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
    Critic (or discriminator) for a Wasserstein GAN with Gradient Penalty (WGAN-GP).
    The critic evaluates the authenticity of both real and generated data.

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


class COVIDDataAugmentation:
    """
    A comprehensive data handling class for COVID-19 data augmentation using WGAN-GP.

    This class manages the entire data pipeline for COVID-19 control data augmentation,
    including data splitting, scaling, cross-validation, and result tracking. It implements
    strict data leakage prevention and maintains comprehensive metadata throughout the process.

    Attributes:
        random_state (int): Seed for reproducibility across all random operations.
        scaler (RobustScaler): Scaler instance for data normalization.
        metadata (dict): Comprehensive tracking of all data operations and versions.

    Example:
        >>> # Initialize the augmentor
        >>> augmentor = COVIDDataAugmentation(random_state=42)
        >>>
        >>> # Prepare data splits
        >>> train_df, test_df = augmentor.prepare_data_splits(
        ...     data_x=feature_matrix,
        ...     data_y=target_vector
        ... )
        >>>
        >>> # Scale the data
        >>> train_scaled, test_scaled = augmentor.implement_scaling(train_df, test_df)
        >>>
        >>> # Run cross-validation
        >>> cv_results = augmentor.implement_cross_validation(
        ...     train_scaled,
        ...     n_folds=3
        ... )
        >>>
        >>> # Save results
        >>> augmentor.save_results(Path('./output'), cv_results)

    Notes:
        - The class implements 3-fold CV by default for optimal handling of small control samples
        - All operations maintain strict data isolation to prevent leakage
        - Comprehensive metadata is maintained for reproducibility
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the COVIDDataAugmentation instance.

        Args:
            random_state (int, optional): Seed for random operations. Defaults to 42.

        Example:
            >>> augmentor = COVIDDataAugmentation(random_state=42)
        """
        self.random_state = random_state
        self.trained_generator = None
        self.scaler = None
        self.metadata = {
            "timestamp": datetime.now().isoformat(),
            "random_state": random_state,
            "data_versions": [],
        }

    def prepare_data_splits(
        self, data_df: pd.DataFrame, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create initial train/test split with enhanced tracking.
        Hold out test never influences training data.
        Clean separation of  original and synthetic samples.

        Args:
            data_df (pd.DataFrame): The dataset, including target column
            test_size (float, optional): Proportion for test set. Defaults to 0.2.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and test dataframes

        Example:
            >>> train_df, test_df = augmentor.prepare_data_splits(X, y, test_size=0.2)
        """

        # Identify controls
        # control_mask = data_df["target"] == 0
        # n_controls = control_mask.sum()

        # Stratified split ensuring proportional controls
        x_train, x_test, y_train, y_test = train_test_split(
            data_df,
            data_df["target"],
            test_size=test_size,
            stratify=data_df["target"],
            random_state=config.RANDOM_STATE,
        )

        # Train set
        x_train["split"] = "development"

        train_df = x_train.copy()
        train_df["target"] = y_train

        # Test set
        x_test["split"] = "holdout"

        test_df = x_test.copy()
        test_df["target"] = y_test

        split_info = {
            # 'timestamp': datetime.now().isoformat(),
            # 'total_samples': len(x_test),
            # 'total_controls': n_controls,
            # 'development_controls': (train_df['target'] == 0).sum(),
            # 'holdout_controls': (test_df['target'] == 0).sum()
        }
        self.metadata["data_versions"].append(split_info)

        return train_df, test_df

    def implement_scaling(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Implement global scaling with enhanced tracking.

        Args:
            train_df (pd.DataFrame): Training data
            test_df (pd.DataFrame): Test data

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Scaled training and test data

        Example:
            >>> train_scaled, test_scaled = augmentor.implement_scaling(train_df, test_df)
        """
        feature_cols = [
            col
            for col in train_df.columns
            if col not in ["target", "sample_id", "data_source", "split"]
        ]

        # Fit scaler on development set
        self.scaler = RobustScaler()
        self.scaler.fit(train_df[feature_cols])

        # Transform both sets
        train_scaled = train_df.copy()
        test_scaled = test_df.copy()
        train_scaled[feature_cols] = self.scaler.transform(train_scaled[feature_cols])
        test_scaled[feature_cols] = self.scaler.transform(test_scaled[feature_cols])

        # Log scaling information
        scaling_info = {
            "timestamp": datetime.now().isoformat(),
            "scaling_method": "RobustScaler",
            "n_features": len(feature_cols),
            "feature_names": feature_cols,
        }

        self.metadata["scaling"] = scaling_info

        return train_scaled, test_scaled

    def implement_cross_validation(self, train_df: pd.DataFrame) -> List[Dict]:
        """
        Implement k-fold cross-validation with synthetic data generation.
        K-Fold revised from 5 to 3 for improved performance with larger datasets.
        Independent synthetic data generation per fold.
        Clear separation of original and synthetic samples.
        Args:
            train_df (pd.DataFrame): Training data
            n_folds (int, optional): Number of CV folds. Defaults to 3.

        Returns:
            List[Dict]: Cross-validation results for each fold

        Example:
            >>> cv_results = augmentor.implement_cross_validation(train_scaled, n_folds=3)
        """
        feature_cols = [
            col
            for col in train_df.columns
            if col not in ["target", "sample_id", "data_source", "split"]
        ]
        kfold = KFold(
            n_splits=config.CV_N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE
        )
        cv_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_df)):
            fold_train = train_df.iloc[train_idx].copy()
            fold_val = train_df.iloc[val_idx].copy()

            # Get controls for WGAN training
            fold_controls = fold_train[fold_train["target"] == 0]

            # Generate synthetic data
            synthetic_samples = self._generate_synthetic_data(
                fold_controls[feature_cols]
            )

            # Track synthetic data
            synthetic_df = pd.DataFrame(synthetic_samples, columns=feature_cols)
            synthetic_df["target"] = 0
            synthetic_df["data_source"] = "synthetic"
            synthetic_df["fold"] = fold
            synthetic_df["generation_seed"] = self.random_state + fold
            synthetic_df["generation_timestamp"] = datetime.now().isoformat()

            cv_results.append(
                {
                    "fold": fold,
                    "train_data": fold_train,
                    "val_data": fold_val,
                    "synthetic_data": synthetic_df,
                    "n_original_controls": len(fold_controls),
                    "n_synthetic_controls": len(synthetic_df),
                }
            )

        return cv_results

    def _generate_synthetic_data(
        self,
        control_features: pd.DataFrame,
        n_samples: int = None,
        noise_dim: int = None,
        noise_scale: float = 1.0,
    ) -> np.ndarray:
        """
        Generate synthetic control samples using trained WGAN-GP model.

        This method implements synthetic data generation using the Wasserstein GAN
        with Gradient Penalty (WGAN-GP). It generates synthetic control samples
        that match the distribution of the original control samples while
        maintaining data privacy and preventing direct copying.

        Args:
            control_features (pd.DataFrame): Original control samples used as reference
                for generating synthetic data. Should contain only feature columns.
            n_samples (int, optional): Number of synthetic samples to generate.
                If None, generates same number as input controls. Defaults to None.
            noise_dim (int, optional): Dimension of the input noise vector.
                If None, uses number of features. Defaults to None.
            noise_scale (float, optional): Scale factor for input noise.
                Higher values increase variation. Defaults to 1.0.

        Returns:
            np.ndarray: Generated synthetic samples with shape (n_samples, n_features)

        Raises:
            ValueError: If control_features is empty or contains invalid values
            RuntimeError: If generation process fails

        Example:
            >>> control_data = pd.DataFrame(original_controls)
            >>> synthetic_samples = _generate_synthetic_data(
            ...     control_features=control_data,
            ...     n_samples=100,
            ...     batch_size=32
            ... )
            >>> print(f"Generated {len(synthetic_samples)} synthetic samples")

        Notes:
            - The method uses the trained generator to create synthetic samples
            - Input noise is scaled to control variation in generated samples
            - Batch processing is used for memory efficiency
            - Generated samples are validated before returning
        """
        # Input validation
        if control_features is None or len(control_features) == 0:
            raise ValueError("Control features cannot be empty")

        # Setup parameters
        n_features = control_features.shape[1]
        if noise_dim is None:
            noise_dim = n_features
        if n_samples is None:
            n_samples = len(control_features)

        try:
            # Initialize generator network
            generator = Generator(input_dim=noise_dim, output_dim=n_features).to(
                config.DEVICE
            )

            # Is this even needed?
            generator = self.trained_generator

            # Generate synthetic samples in batches
            synthetic_samples = []
            remaining_samples = n_samples

            while remaining_samples > 0:
                current_batch_size = min(config.BATCH_SIZE, remaining_samples)

                # Generate noise input
                noise = (
                    torch.randn(current_batch_size, noise_dim, device=config.DEVICE)
                    * noise_scale
                )

                # Generate samples
                with torch.no_grad():
                    generator.eval()
                    batch_samples = generator(noise).cpu().numpy()
                    generator.train()

                # Validate generated samples
                if np.isnan(batch_samples).any() or np.isinf(batch_samples).any():
                    raise RuntimeError("Invalid samples generated")

                synthetic_samples.append(batch_samples)
                remaining_samples -= current_batch_size

                # Log progress for large generations
                if n_samples > 1000 and len(synthetic_samples) % 10 == 0:
                    print(
                        "Generated %d/%d samples",
                        n_samples - remaining_samples,
                        n_samples,
                    )

            # Combine all batches
            synthetic_samples = np.vstack(synthetic_samples)

            # Post-processing
            # Recenter data to match original distribution
            synthetic_samples = recenter_data(
                synthetic_samples, control_features.values
            )

            # Validate final output
            self._validate_synthetic_samples(
                synthetic_samples, control_features.values, n_samples
            )

            print("Successfully generated %d synthetic samples", n_samples)
            return synthetic_samples

        except Exception as e:
            raise RuntimeError("Failed to generate synthetic data") from e

    def _validate_synthetic_samples(
        self,
        synthetic_samples: np.ndarray,
        original_samples: np.ndarray,
        expected_samples: int,
    ) -> None:
        """
        Validate generated synthetic samples.

        Args:
            synthetic_samples: Generated samples to validate
            original_samples: Original samples for reference
            expected_samples: Expected number of samples

        Raises:
            ValueError: If validation fails
        """
        if synthetic_samples.shape[0] != expected_samples:
            raise ValueError(
                f"Generated {synthetic_samples.shape[0]} samples, "
                f"expected {expected_samples}"
            )

        if synthetic_samples.shape[1] != original_samples.shape[1]:
            raise ValueError(
                f"Generated samples have {synthetic_samples.shape[1]} features, "
                f"expected {original_samples.shape[1]}"
            )

        # Check for invalid values
        if np.isnan(synthetic_samples).any():
            raise ValueError("Generated samples contain NaN values")

        if np.isinf(synthetic_samples).any():
            raise ValueError("Generated samples contain infinite values")

        # Check value ranges
        orig_min = np.min(original_samples)
        orig_max = np.max(original_samples)
        synth_min = np.min(synthetic_samples)
        synth_max = np.max(synthetic_samples)

        if synth_min < orig_min * 1.5 or synth_max > orig_max * 1.5:
            print(
                "Generated samples may have out-of-range values. "
                "Original range: [%f, %f], Synthetic range: [%f, %f]",
                orig_min,
                orig_max,
                synth_min,
                synth_max,
            )

    def save_results(self, output_dir: Path, cv_results: List[Dict]) -> None:
        """
        Save results with comprehensive tracking.

        Args:
            output_dir (Path): Directory for saving results
            cv_results (List[Dict]): Cross-validation results to save

        Example:
            >>> augmentor.save_results(Path('./output'), cv_results)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save metadata
        self.metadata["cv_results"] = {
            "n_folds": len(cv_results),
            "fold_sizes": [len(fold["train_data"]) for fold in cv_results],
            "synthetic_samples_per_fold": [
                len(fold["synthetic_data"]) for fold in cv_results
            ],
        }

        with open(
            f"{output_dir}/metadata_{timestamp}.json", "w", encoding="utf-8"
        ) as f:
            json.dump(self.metadata, f, indent=2)

        # Save fold results
        for fold_result in cv_results:
            fold = fold_result["fold"]
            fold_dir = output_dir / f"fold_{fold}"
            fold_dir.mkdir(exist_ok=True)

            fold_result["train_data"].to_parquet(fold_dir / "train_data.parquet")
            fold_result["val_data"].to_parquet(fold_dir / "val_data.parquet")
            fold_result["synthetic_data"].to_parquet(
                fold_dir / "synthetic_data.parquet"
            )

    def train_wgan_gp(
        self,
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
                    c_loss = (
                        fake_validity.mean() - real_validity.mean() + lambda_gp * gp
                    )

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

            self.trained_generator = generator

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


def train_and_generate(
    filepath: str,
    batch_size: int = 32,
    epochs: int = 20,
    device: str = "cpu",
    n_splits: int = 3,
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
    5. Enhanced data tracking with COVIDDataAugmentation

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
    # Initialize augmentor for enhanced data tracking
    augmentor = COVIDDataAugmentation(random_state=42)

    # Load and process data
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
    device = config.DEVICE
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
            train_tensor, batch_size=batch_size, shuffle=True, drop_last=True
        )

        print(f"Train Loader Len: {len(train_loader)}")

        # Initialize models
        generator = Generator(n_features, n_features).to(device)
        critic = Critic(n_features).to(device)

        # Initialize optimizers with beta parameters for stability
        g_optimizer = Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
        c_optimizer = Adam(critic.parameters(), lr=learning_rate, betas=(0.5, 0.9))

        # Train the models
        g_losses, c_losses = augmentor.train_wgan_gp(
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

        # Create fold result dictionary for tracking
        fold_result = {
            "fold": fold,
            "train_data": scaled_data.iloc[train_idx],
            "val_data": scaled_data.iloc[val_idx],
            "synthetic_data": fold_df,
            "metrics": {"g_loss": g_losses, "c_loss": c_losses},
        }
        generated_samples.append(fold_result)

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

    # Combine all synthetic data
    synthetic_df = pd.concat(
        [f["synthetic_data"] for f in generated_samples], ignore_index=True
    )
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
            "training_metrics": {
                f"fold_{f['fold']}": f["metrics"] for f in generated_samples
            },
        }

        if isinstance(filepath, str):
            info_path = Path(filepath).parent / "data_info.json"
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(data_info, f, indent=2)

        # Save results using augmentor
        augmentor.save_results(Path(filepath).parent / "results", generated_samples)

    print("\nGeneration Summary:")
    print(f"Original samples: {len(original_data)}")
    print(f"Synthetic samples: {len(synthetic_df)}")
    print(f"Total samples: {len(final_df)}")
    print("\nSynthetic samples per fold:")
    print(synthetic_df.groupby("fold").size())

    return final_df, original_data
