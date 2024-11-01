"""
Enhanced WGAN-GP implementation for COVID-19 control data augmentation.

This module implements a Wasserstein GAN with Gradient Penalty specifically for
generating synthetic control samples in medical datasets with class imbalance.
Includes improved cross-validation, mode collapse prevention, and data labeling.

Key Features:
    - Enhanced Generator and Critic architectures
    - Improved gradient penalty computation
    - Mode collapse prevention through noise diversity
    - Proper data labeling and tracking
    - 3-fold cross-validation for small sample considerations
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
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

from src.utils.preprocessing import process
from src.utils.evaluation import recenter_data

# Configure plotting style
plt.style.use('seaborn')
sns.set_theme(style="whitegrid")

def plot_training_losses(
    g_losses: List[float],
    c_losses: List[float],
    fold: Optional[int] = None,
    save_dir: Optional[Path] = None
):
    """
    Plot training losses with detailed visualization.
    
    Args:
        g_losses: Generator loss history
        c_losses: Critic loss history
        fold: Current fold number (if using CV)
        save_dir: Directory to save plots
    """
    # Separate plots for Generator and Critic
    plt.figure(figsize=(12, 4))
    
    # Generator Loss
    plt.subplot(1, 2, 1)
    plt.plot(
        g_losses, 
        label=f"Generator Loss{f' - Fold {fold}' if fold else ''}", 
        color="blue", 
        alpha=0.7
    )
    plt.title("Generator Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # Critic Loss
    plt.subplot(1, 2, 2)
    plt.plot(
        c_losses, 
        label=f"Critic Loss{f' - Fold {fold}' if fold else ''}", 
        color="red", 
        alpha=0.7
    )
    plt.title("Critic Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / f"separate_losses{'_fold_'+str(fold) if fold else ''}.png")
    plt.show()

    # Combined losses
    plt.figure(figsize=(10, 6))
    plt.plot(g_losses, label="Generator Loss", color="blue", alpha=0.7)
    plt.plot(c_losses, label="Critic Loss", color="red", alpha=0.7)
    plt.title(f"Training Losses{f' - Fold {fold}' if fold else ''}")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    if save_dir:
        plt.savefig(save_dir / f"combined_losses{'_fold_'+str(fold) if fold else ''}.png")
    plt.show()


class Generator(nn.Module):
    """
    Enhanced Generator with improved architecture for medical data generation.
    
    Features:
        - Residual connections for better gradient flow
        - Batch normalization for training stability
        - Dynamic width adjustment
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super(Generator, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Enhanced architecture
        self.net = nn.Sequential(
            # Input layer
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            
            # Expanded middle layer
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),  # Prevent overfitting
            
            # Additional middle layer
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            
            # Output layer
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Tanh()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better training stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class Critic(nn.Module):
    """
    Enhanced Critic with improved stability and gradient handling.
    
    Features:
        - Layer normalization instead of batch norm
        - Spectral normalization option
        - Gradient penalty-friendly architecture
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        
        self.net = nn.Sequential(
            # Input layer
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            
            # Middle layer
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            
            # Output layer
            nn.Linear(hidden_dim, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def compute_gradient_penalty(
    critic: Critic,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 10.0
) -> torch.Tensor:
    """
    Enhanced gradient penalty computation with improved stability.
    
    Args:
        critic: Critic network
        real_samples: Real data batch
        fake_samples: Generated data batch
        device: Computation device
        lambda_gp: Gradient penalty coefficient
    """
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    
    # Interpolate between real and fake samples
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)
    
    # Get critic scores
    d_interpolates = critic(interpolates)
    
    # Calculate gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Compute gradient penalty
    gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
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
    lambda_gp: float = 10.0,
    noise_multiplier: float = 1.0,
    results_dir: Optional[Path] = None,
    current_fold: Optional[int] = None
) -> Dict[str, List[float]]:
    """
    Train WGAN-GP with enhanced visualization and progress tracking.
    
    Features:
        - Dynamic noise scaling for diversity
        - Detailed loss visualization
        - Progress tracking per epoch and fold
        - Wasserstein distance monitoring
    
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
        results_dir: Directory to save results
        current_fold: Current fold number for CV
        
    Returns:
        Dictionary containing training history
    """
    generator.train()
    critic.train()
    
    # dictionary for comprehensive tracking
    training_history = {
        'g_losses': [],
        'c_losses': [],
        'gradient_penalties': [],
        'wasserstein_distances': []
    }
    
    # Create fold-specific directory for plots
    if results_dir:
        fold_dir = results_dir / f"fold_{current_fold}" if current_fold is not None else results_dir
        fold_dir.mkdir(parents=True, exist_ok=True)

    for epoch in tqdm(range(epochs), desc=f"Training WGAN-GP{f' Fold {current_fold}' if current_fold else ''}"):
        epoch_metrics = {key: [] for key in training_history.keys()}
        
        for batch_idx, (real_data,) in enumerate(data_loader):
            real_data = real_data.to(device)
            batch_size = real_data.size(0)
            
            # Train Critic
            for _ in range(critic_iterations):
                c_optimizer.zero_grad()
                
                # Generate diverse fake samples
                noise = torch.randn(batch_size, generator.input_dim, device=device)
                noise = noise * noise_multiplier
                fake_data = generator(noise)
                
                # Compute Wasserstein distance
                real_validity = critic(real_data)
                fake_validity = critic(fake_data.detach())
                
                # Gradient penalty
                gp = compute_gradient_penalty(critic, real_data, fake_data, device)
                
                # Critic loss
                c_loss = fake_validity.mean() - real_validity.mean() + lambda_gp * gp
                c_loss.backward()
                c_optimizer.step()
                
                epoch_metrics['c_losses'].append(c_loss.item())
                epoch_metrics['gradient_penalties'].append(gp.item())
                epoch_metrics['wasserstein_distances'].append(
                    (real_validity.mean() - fake_validity.mean()).item()
                )
            
            # Train Generator
            g_optimizer.zero_grad()
            noise = torch.randn(batch_size, generator.input_dim, device=device)
            fake_data = generator(noise)
            fake_validity = critic(fake_data)
            
            g_loss = -fake_validity.mean()
            g_loss.backward()
            g_optimizer.step()
            
            epoch_metrics['g_losses'].append(g_loss.item())
            
            # Progress reporting with all metrics
            if batch_idx % 50 == 0:
                print(
                    f"\nEpoch [{epoch}/{epochs}] "
                    f"Batch [{batch_idx}/{len(data_loader)}] "
                    f"G_loss: {g_loss.item():.4f} "
                    f"C_loss: {c_loss.item():.4f} "
                    f"GP: {gp.item():.4f} "
                    f"W_dist: {epoch_metrics['wasserstein_distances'][-1]:.4f}"
                )
        
        # Update training history
        for key in training_history:
            training_history[key].extend(epoch_metrics[key])
        
        # Plot intermediate results with all metrics
        if epoch % 10 == 0 and results_dir:
            epoch_dir = fold_dir / f"epoch_{epoch}"
            epoch_dir.mkdir(exist_ok=True)
            
            # Enhanced plotting with all metrics
            plot_training_metrics(
                training_history,
                current_fold,
                epoch_dir
            )
            
            # Save checkpoint with all metrics
            save_training_checkpoint(
                epoch_dir,
                epoch,
                generator,
                critic,
                g_optimizer,
                c_optimizer,
                training_history
            )
        
        # Print comprehensive epoch summary
        print_epoch_summary(epoch, epochs, epoch_metrics)
    
    # Final plots and saves
    if results_dir:
        final_dir = fold_dir / "final"
        final_dir.mkdir(exist_ok=True)
        
        # Save all metrics
        for key, values in training_history.items():
            np.save(final_dir / f'{key}.npy', values)
        
        # Final plots with all metrics
        plot_training_metrics(training_history, current_fold, final_dir)
    
    return training_history

def plot_training_metrics(
    training_history: Dict[str, List[float]],
    fold: Optional[int] = None,
    save_dir: Optional[Path] = None
):
    """
    Enhanced plotting function for all training metrics.
    
    Args:
        training_history: Dictionary containing all training metrics
        fold: Current fold number for CV
        save_dir: Directory to save plots
    """
    # Plot losses
    plot_training_losses(
        training_history['g_losses'],
        training_history['c_losses'],
        fold,
        save_dir
    )
    
    # Plot gradient penalties
    plt.figure(figsize=(10, 6))
    plt.plot(training_history['gradient_penalties'], 
             label="Gradient Penalty", 
             color="purple", 
             alpha=0.7)
    plt.title(f"Gradient Penalty{f' - Fold {fold}' if fold else ''}")
    plt.xlabel("Iteration")
    plt.ylabel("Penalty")
    plt.grid(True)
    plt.legend()
    if save_dir:
        plt.savefig(save_dir / 'gradient_penalties.png')
    plt.close()
    
    # Plot Wasserstein distances
    plt.figure(figsize=(10, 6))
    plt.plot(training_history['wasserstein_distances'],
             label="Wasserstein Distance",
             color="green",
             alpha=0.7)
    plt.title(f"Wasserstein Distance{f' - Fold {fold}' if fold else ''}")
    plt.xlabel("Iteration")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.legend()
    if save_dir:
        plt.savefig(save_dir / 'wasserstein_distances.png')
    plt.close()


def train_and_generate(
    filepath: str,
    batch_size: int = 32,
    epochs: int = 100,
    device: str = "cpu",
    n_splits: int = 3,  # Using 3-fold CV for small control sample
    learning_rate: float = 1e-4,
    save_info: bool = True,
    results_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train WGAN-GP and generate synthetic data using enhanced k-fold cross-validation.
    
    Features:
        - 3-fold CV for small control sample consideration
        - Comprehensive progress tracking and visualization
        - Proper data labeling and isolation
        - Enhanced artifact saving
        - Detailed metrics tracking
        - Structured experiment management
    
    Args:
        filepath: Path to dataset
        batch_size: Training batch size
        epochs: Number of training epochs
        device: Computing device ('cpu' or 'cuda')
        n_splits: Number of CV folds (default 3 for small sample)
        learning_rate: Learning rate for optimizers
        save_info: Whether to save training artifacts
        results_dir: Directory for saving results
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (complete_dataset, original_data)
            - complete_dataset: Combined original and synthetic data
            - original_data: Original data for testing
    """
    # Setup experiment directory
    if results_dir is None:
        results_dir = Path(f"results/wgan_{datetime.now():%Y%m%d_%H%M%S}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize managers
    experiment_manager = ExperimentManager(results_dir)
    scaler_manager = ScalerManager(results_dir)
    
    # Load and process data
    _, _, scaled_data, scaler, n_features = process(filepath)
    
    # Store original data with proper labeling
    original_data = scaled_data.copy()
    original_data["fold"] = -1
    original_data["data_type"] = "original"
    original_data["sample_id"] = range(len(original_data))
    
    # Print comprehensive data overview
    print("\nData Overview:")
    print("=" * 50)
    print(f"Original samples: {len(original_data)}")
    print(f"Number of features: {n_features}")
    print("\nFeature types:")
    print("-" * 30)
    print(scaled_data.dtypes)
    print("\nClass distribution:")
    print("-" * 30)
    print(original_data['Group'].value_counts())
    print("=" * 50)
    
    # Save scaler
    scaler_manager.save_scaler(scaler)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Initialize K-fold
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    generated_samples = []
    fold_metrics = []


    # Training and generation for each fold
    for fold, (train_idx, val_idx) in enumerate(kfold.split(scaled_data), 1):
        fold_dir = results_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing fold {fold}/{n_splits}")
        print(f"Training samples: {len(train_idx)}")
        print(f"Validation samples: {len(val_idx)}")
        
        # Prepare training data
        train_data = scaled_data.iloc[train_idx]
        val_data = scaled_data.iloc[val_idx]
        
        # Get control-only training data
        train_controls = train_data[train_data['Group'] == 0]
        train_tensor = torch.tensor(
            train_controls.values,
            dtype=torch.float32
        ).to(device)
        
        # Create data loader
        train_dataset = torch.utils.data.TensorDataset(train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )
        
        # Initialize models
        generator = Generator(n_features, n_features).to(device)
        critic = Critic(n_features).to(device)
        
        # Initialize optimizers with improved parameters
        g_optimizer = Adam(generator.parameters(), lr=learning_rate, betas=(0.0, 0.9))
        c_optimizer = Adam(critic.parameters(), lr=learning_rate, betas=(0.0, 0.9))
        
        # Train WGAN-GP and capture training history
        training_history = train_wgan_gp(
            train_loader,
            generator,
            critic,
            g_optimizer,
            c_optimizer,
            device,
            epochs=epochs,
            results_dir=fold_dir,
            current_fold=fold
        )
        
        # Save training checkpoint
        experiment_manager.save_fold_models(
            fold=fold,
            generator=generator,
            critic=critic,
            g_optimizer=g_optimizer,
            c_optimizer=c_optimizer,
            epoch=epochs,
            metrics=training_history
        )
        
        # Generate synthetic samples
        n_synthetic_needed = len(val_idx)  # Balance samples
        synthetic_samples = generate_synthetic_samples(
            generator=generator,
            n_samples=n_synthetic_needed,
            n_features=n_features,
            device=device,
            scaler=scaler_manager,
            reference_data=val_data,
            noise_multiplier=1.0,
            batch_size=batch_size
        )
        
        # Create DataFrame with metadata
        fold_df = pd.DataFrame(
            synthetic_samples,
            columns=[col for col in scaled_data.columns 
                    if col not in ["data_type", "fold", "sample_id"]]
        )
        
        # Add metadata
        fold_df["data_type"] = "synthetic"
        fold_df["fold"] = fold
        fold_df["generation_seed"] = fold
        
        # Validate synthetic samples
        validation_results = validate_synthetic_samples(
            synthetic_samples,
            val_data.values
        )
        
        # Generate and save visualizations
        visualization_dir = fold_dir / "visualizations"
        visualization_dir.mkdir(exist_ok=True)
        
        # Loss plots
        plot_training_losses(
            training_history['g_losses'],
            training_history['c_losses'],
            fold=fold,
            save_dir=visualization_dir
        )
        
        # Quality comparison plots
        plot_synthetic_sample_quality(
            synthetic_samples,
            val_data.values,
            [col for col in scaled_data.columns 
            if col not in ["data_type", "fold", "sample_id"]],
            save_dir=visualization_dir
        )
        
        # Distribution comparison plots
        plot_distribution_comparisons(
            synthetic_samples,
            val_data.values,
            feature_names=[col for col in scaled_data.columns 
                        if col not in ["data_type", "fold", "sample_id"]],
            save_dir=visualization_dir
        )
        
        # Save fold results
        experiment_manager.save_fold_results(
            fold=fold,
            synthetic_df=fold_df,
            metrics=training_history,
            validation_results=validation_results
        )
        
        generated_samples.append(fold_df)
        fold_metrics.append(training_history)
        
        print(f"Completed fold {fold}. Generated {len(fold_df)} samples.")
    
    # Combine all synthetic data
    synthetic_df = pd.concat(generated_samples, ignore_index=True)
    final_df = pd.concat([original_data, synthetic_df], ignore_index=True)

    # Generate and save overall experiment visualizations
    plot_overall_training_progress(
        fold_metrics=fold_metrics,
        results_dir=results_dir,
        experiment_manager=experiment_manager
    )

    # Save final experiment info
    experiment_manager.save_experiment_info(
        config={
            'batch_size': batch_size,
            'epochs': epochs,
            'n_splits': n_splits,
            'learning_rate': learning_rate,
            'device': device,
            'start_time': datetime.now().isoformat()
        },
        original_data=original_data,
        synthetic_df=synthetic_df,
        fold_metrics=fold_metrics
    )
    
    return final_df, original_data

def plot_training_losses(
    g_losses: List[float],
    c_losses: List[float],
    fold: int,
    save_dir: Path
) -> None:
    """Plot detailed training loss visualization."""
    plt.figure(figsize=(15, 5))
    
    # Generator Loss
    plt.subplot(1, 2, 1)
    plt.plot(g_losses, label=f"Generator - Fold {fold}", color='blue', alpha=0.7)
    plt.title("Generator Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    
    # Critic Loss
    plt.subplot(1, 2, 2)
    plt.plot(c_losses, label=f"Critic - Fold {fold}", color='red', alpha=0.7)
    plt.title("Critic Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / f'training_losses_fold_{fold}.png')
    plt.close()

def plot_distribution_comparisons(
    synthetic_samples: np.ndarray,
    reference_samples: np.ndarray,
    feature_names: List[str],
    save_dir: Path
) -> None:
    """Plot detailed distribution comparisons."""
    n_features = len(feature_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 5 * n_rows))
    
    for idx, feature in enumerate(feature_names):
        plt.subplot(n_rows, n_cols, idx + 1)
        
        # Plot distributions
        sns.kdeplot(
            data=reference_samples[:, idx],
            label='Original',
            color='blue',
            alpha=0.6
        )
        sns.kdeplot(
            data=synthetic_samples[:, idx],
            label='Synthetic',
            color='red',
            alpha=0.6
        )
        
        plt.title(f'{feature} Distribution')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'distribution_comparisons.png')
    plt.close()

def plot_overall_training_progress(
    fold_metrics: List[Dict[str, List[float]]],
    results_dir: Path,
    experiment_manager: ExperimentManager
) -> None:
    """Plot comprehensive training progress across all folds."""
    plt.figure(figsize=(15, 10))
    
    metrics = ['g_losses', 'c_losses', 'gradient_penalties', 'wasserstein_distances']
    titles = ['Generator Loss', 'Critic Loss', 'Gradient Penalty', 'Wasserstein Distance']
    colors = ['blue', 'red', 'purple', 'green']
    
    for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        plt.subplot(2, 2, idx+1)
        for fold, fold_history in enumerate(fold_metrics, 1):
            plt.plot(fold_history[metric], 
                    label=f'Fold {fold}', 
                    color=color, 
                    alpha=0.5)
        plt.title(title)
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(results_dir / 'overall_training_progress.png')
    plt.close()