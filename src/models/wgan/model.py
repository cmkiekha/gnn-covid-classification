# src/models/data_augmentation_v2/GAN.py
import torch
from torch import nn
import logging
from typing import Dict, List, Optional

# Update imports to use absolute imports
from src.utils.enhanced.preprocessing import DataProcessor
from src.validation.metrics_tracking import MetricsTracker
from src.utils.enhanced.evaluation import evaluate_generation


class Generator(nn.Module):
    """
    Generator network for WGAN-GP implementation.

    Architecture:
    - Multiple hidden layers with LayerNorm and LeakyReLU
    - Dropout for regularization
    - Tanh activation in output layer

    Args:
        input_dim (int): Dimension of input noise vector
        hidden_dims (List[int]): List of hidden layer dimensions
    """


    def __init__(self, input_dim: int, hidden_dims: List[int]):
        """Initialize Generator."""
        super().__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.1),
                ]
            )
            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, input_dim))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)


    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate samples from noise.
        
        Args:
            z: Input noise tensor
            
        Returns:
            torch.Tensor: Generated samples
        """
        return self.model(z)


class Critic(nn.Module):
    """
    Critic network for WGAN-GP implementation.  
This implementation includes:
1. Complete Generator and Critic architectures
2. Full training loop with gradient penalty
3. K-fold cross-validation
4. Comprehensive metrics tracking
5. Visualization integration
6. Sample generation functionality

    Architecture:
    - Multiple hidden layers with LayerNorm and LeakyReLU
    - Dropout for regularization
    - Linear output for Wasserstein distance estimation

    Args:
        input_dim (int): Dimension of input data
        hidden_dims (List[int]): List of hidden layer dimensions
    """

    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super(Critic, self).__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.1),
                ]
            )
            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class WGANGPTrainer:
    """
    WGAN-GP trainer with k-fold cross-validation support.

    Features:
    - K-fold cross-validation
    - Gradient penalty computation
    - Metrics tracking
    - Visualization support
    - Sample generation

    Args:
        config (Dict): Configuration dictionary containing model and training parameters
    """

    def __init__(self, config: Dict):
        self.config = config
        self.metrics_tracker = MetricsTracker()
        self.data_processor = DataProcessor()
        self.training_visualizer = TrainingVisualizer()
        self.evaluation_visualizer = EvaluationVisualizer()
        self._setup_logging()

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model parameters
        self.input_dim = config["model"]["input_dim"]
        self.hidden_dims = config["model"]["hidden_dims"]
        self.lr = config["training"]["learning_rate"]
        self.beta1 = config["training"]["beta1"]
        self.beta2 = config["training"]["beta2"]
        self.n_critic = config["training"]["n_critic"]
        self.lambda_gp = config["training"]["lambda_gp"]

        # Initialize models
        self.generator = Generator(self.input_dim, self.hidden_dims).to(self.device)
        self.critic = Critic(self.input_dim, self.hidden_dims).to(self.device)

        # Initialize optimizers
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
        )
        self.c_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
        )

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Compute gradient penalty for WGAN-GP"""
        alpha = torch.rand((real_samples.size(0), 1), device=self.device)
        interpolates = (
            alpha * real_samples + (1 - alpha) * fake_samples
        ).requires_grad_(True)

        d_interpolates = self.critic(interpolates)
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_with_kfold(self, control_data, n_folds=5):
        """Detailed k-fold cross-validation implementation."""
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(control_data)):
            logging.info(f"Training fold {fold + 1}/{n_folds}")

            # Split data
            train_data = control_data[train_idx]
            val_data = control_data[val_idx]

            # Train fold
            train_metrics = self._train_fold(train_data)

            # Validate fold
            val_metrics = self._validate_fold(val_data)

            # Update metrics for this fold
            self.metrics_tracker.update_fold_metrics(
                fold,
                {
                    "generator_loss": train_metrics["generator_loss"],
                    "critic_loss": train_metrics["critic_loss"],
                    "validation_metrics": val_metrics,
                },
            )

            # Plot training progress for this fold
            self.training_visualizer.plot_losses(
                train_metrics["generator_loss"], train_metrics["critic_loss"]
            )

        # Get cross-validation summary
        cv_summary = self.metrics_tracker.get_cross_validation_summary()

        # Plot final evaluation results
        self.evaluation_visualizer.plot_metrics(cv_summary)

        return cv_summary

    def _train_fold(self, fold_data):
        """Train WGAN-GP on a single fold"""
        epoch_metrics = {
            "generator_loss": [],
            "critic_loss": [],
            "gradient_penalty": [],
        }

        dataloader = torch.utils.data.DataLoader(
            fold_data, batch_size=self.config["training"]["batch_size"], shuffle=True
        )

        for epoch in range(self.config["training"]["epochs"]):
            for batch_idx, real_samples in enumerate(dataloader):
                batch_size = real_samples.size(0)
                real_samples = real_samples.to(self.device)

                # Train Critic
                self.c_optimizer.zero_grad()

                # Generate fake samples
                z = torch.randn(batch_size, self.input_dim, device=self.device)
                fake_samples = self.generator(z)

                # Compute critic losses
                real_validity = self.critic(real_samples)
                fake_validity = self.critic(fake_samples.detach())
                gradient_penalty = self.compute_gradient_penalty(
                    real_samples.data, fake_samples.data
                )

                c_loss = (
                    -torch.mean(real_validity)
                    + torch.mean(fake_validity)
                    + self.lambda_gp * gradient_penalty
                )
                c_loss.backward()
                self.c_optimizer.step()

                # Train Generator every n_critic steps
                if batch_idx % self.n_critic == 0:
                    self.g_optimizer.zero_grad()

                    # Generate fake samples
                    z = torch.randn(batch_size, self.input_dim, device=self.device)
                    fake_samples = self.generator(z)
                    fake_validity = self.critic(fake_samples)

                    g_loss = -torch.mean(fake_validity)
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Record metrics
                    epoch_metrics["generator_loss"].append(g_loss.item())
                    epoch_metrics["critic_loss"].append(c_loss.item())
                    epoch_metrics["gradient_penalty"].append(gradient_penalty.item())

            # Log epoch progress
            if (epoch + 1) % 10 == 0:
                logging.info(
                    f"Epoch [{epoch+1}/{self.config['training']['epochs']}] "
                    f"G_loss: {g_loss.item():.4f} "
                    f"C_loss: {c_loss.item():.4f}"
                )

        # Update training metrics
        self.metrics_tracker.update_training_metrics(epoch_metrics)

        return epoch_metrics

    def _validate_fold(self, validation_data):
        """Validate on held-out fold data"""
        self.generator.eval()

        with torch.no_grad():
            # Generate samples
            generated_samples = self.generate_samples(len(validation_data))

            # Evaluate samples
            validation_metrics = evaluate_generation(validation_data, generated_samples)

            # Update generation metrics
            self.metrics_tracker.update_generation_metrics(validation_metrics)

            # Plot validation results
            self.evaluation_visualizer.plot_distributions(
                validation_data, generated_samples
            )

        self.generator.train()
        return validation_metrics

    def generate_samples(self, n_samples: int):
        """Generate synthetic samples"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.input_dim, device=self.device)
            samples = self.generator(z)
            samples = samples.cpu().numpy()
        self.generator.train()
        return samples
