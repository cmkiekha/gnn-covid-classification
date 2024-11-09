# REVISED GAN.PY WITH INCLUSION OF `train_wgan_gp` 11-07-2024

# Import configuration settings
from src.utils.evaluation_v3 import recenter_data
import config
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple


class Generator(nn.Module):
    """
    Generator network for WGAN-GP, designed to transform random noise into synthetic data samples.

    Parameters:
        input_dim (int): Dimension of the input noise vector.
        output_dim (int): Dimension of the output data, matching the original data dimension.

     CONSIDER ENHANCING THE GENERATOR NETWORK TO IMPROVE DATA SYNTHESIS QUALITY   
    """
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)

    # ORIGNAL GENERATOR NETWORK
    #  def __init__(self, input_dim: int, output_dim: int):
    #     super().__init__()
    #     self.input_dim = input_dim
    #     self.layers = nn.Sequential(
    #         nn.Linear(input_dim, 256),
    #         nn.BatchNorm1d(256),
    #         nn.LeakyReLU(0.2),
    #         nn.Dropout(0.3),
    #         nn.Linear(256, 512),
    #         nn.BatchNorm1d(512),
    #         nn.LeakyReLU(0.2),
    #         nn.Dropout(0.3),
    #         nn.Linear(512, 512),
    #         nn.BatchNorm1d(512),
    #         nn.LeakyReLU(0.2),
    #         nn.Dropout(0.3),
    #         nn.Linear(512, output_dim),
    #         nn.Tanh(),
    #     )

    # def forward(self, z):
    #     return self.layers(z)


class Critic(nn.Module):
    """
    Critic network for WGAN-GP, designed to evaluate the quality of real and synthetic data.

    Parameters:
        input_dim (int): Dimension of the input data, matching the original data dimension.
    """
    def __init__(self, input_dim):
        self.input_dim = input_dim

        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_dim, 512)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x)


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
) -> Tuple[List[float], List[float]]:
    """
    Trains the WGAN-GP model, alternating updates to the Critic and Generator with an emphasis
    on stability and data diversity through the use of a Gradient Penalty (GP) term.

    This function implements the WGAN-GP (Wasserstein GAN with Gradient Penalty) framework,
    which is designed to improve the quality and diversity of generated data while
    ensuring training stability. It operates by alternating between Critic and Generator
    training steps, updating the Critic more frequently than the Generator in each epoch
    to approximate the Wasserstein distance, which helps maintain a smooth and stable training
    process. The gradient penalty, controlled by `lambda_gp`, enforces a soft Lipschitz continuity
    constraint on the Critic's gradients, further stabilizing training.

    Key Mechanisms:
        - **Gradient Penalty (`lambda_gp`)**: A crucial term in the loss function for the Critic,
          enforcing smooth gradients and Lipschitz continuity to stabilize the GAN. This penalty
          discourages sharp gradients by penalizing deviations in the gradient norm from 1.0,
          thus maintaining a stable training environment that allows the Critic to guide the
          Generator toward realistic data synthesis.

        - **Critic Iterations**: The Critic is updated multiple times per Generator update (defined
          by `critic_iterations`). This setup allows the Critic to establish a more accurate
          approximation of the Wasserstein distance, enabling it to provide a stable signal to
          the Generator during training.

        - **Dynamic Noise Scaling**: Fresh noise is injected into the Generator each step,
          encouraging the Generator to explore a broader data space, which enhances the diversity
          of generated samples.

    Args:
        data_loader (DataLoader): DataLoader containing real data samples for training.
        generator (Generator): The Generator network responsible for producing synthetic data samples.
        critic (Critic): The Critic (or Discriminator) network, which evaluates real and generated data samples.
        g_optimizer (Adam): Optimizer for the Generator, facilitating parameter updates.
        c_optimizer (Adam): Optimizer for the Critic, facilitating parameter updates.
        device (torch.device): Specifies the device (CPU/GPU) for computations.
        epochs (int, optional): Number of training epochs to perform. Default is 100.
        critic_iterations (int, optional): Number of Critic updates per Generator update.
            Setting this to a higher value (e.g., 5) helps the Critic accurately assess
            real vs. generated samples, improving the GAN’s stability.
        lambda_gp (float, optional): Coefficient for the Gradient Penalty term, a critical
            hyperparameter for enforcing Lipschitz continuity. Default is 10.0. The value can
            be adjusted in `config.py` for experimentation with training stability and data quality.

    Returns:
        Tuple[List[float], List[float]]: Two lists, containing loss values for the Generator and Critic respectively.
            These losses provide insights into the stability of the GAN during training and help
            monitor the quality of data generation over time.

    How It Works:
        - **Critic Training**: For each real data batch, the Critic is updated several times to better
          approximate the Wasserstein distance. This includes evaluating real and generated samples,
          applying the Gradient Penalty, and calculating the Critic loss.

        - **Gradient Penalty Application**: After evaluating real and generated samples, the gradient
          penalty term is calculated to enforce smooth gradients, which stabilizes training by keeping
          the Critic's gradients close to 1 along the interpolation path between real and generated samples.

        - **Generator Training**: After updating the Critic, the Generator is trained once by minimizing
          the negative output of the Critic on generated samples, guiding it to produce more realistic samples.

    Example Usage:
        >>> g_losses, c_losses = train_wgan_gp(
                data_loader=data_loader,
                generator=generator,
                critic=critic,
                g_optimizer=g_optimizer,
                c_optimizer=c_optimizer,
                device=device,
                epochs=100,
                critic_iterations=5,
                lambda_gp=10.0
            )
        >>> print("Generator Losses:", g_losses)
        >>> print("Critic Losses:", c_losses)

    Notes:
        This function is the core training function for WGAN-GP, controlling the iterative
        process required to produce stable and high-quality synthetic data through carefully
        coordinated updates to both the Generator and Critic.

    """

    generator.train()
    critic.train()

    g_losses, c_losses = [], []

    for epoch in range(epochs):
        epoch_g_loss = 0
        epoch_c_loss = 0
        n_batches = 0

        for real_data in data_loader:
            real_data = real_data.to(device) # Differs from GAN_v2__ recheck
            batch_size = real_data.size(0)

            # Train Critic more frequently
            for _ in range(critic_iterations):
                c_optimizer.zero_grad()
                noise = torch.randn(batch_size, generator.input_dim, device=device)
                fake_samples = generator(noise)

                # Critic scores
                real_scores = critic(real_data)
                fake_scores = critic(fake_samples.detach())

                # Gradient penalty
                gp = compute_gradient_penalty(critic, real_data, fake_samples, device)
                c_loss = fake_scores.mean() - real_scores.mean() + lambda_gp * gp

                c_loss.backward()
                c_optimizer.step()
                epoch_c_loss += c_loss.item()

            # Train Generator
            g_optimizer.zero_grad()
            noise = torch.randn(batch_size, generator.input_dim, device=device)
            fake_samples = generator(noise)
            g_loss = -critic(fake_samples).mean()

            g_loss.backward()
            g_optimizer.step()

            # Track losses
            g_losses.append(g_loss.item())
            c_losses.append(c_loss.item())
            epoch_g_loss += g_loss.item()
            n_batches += 1

        print(
            f"Epoch {epoch+1}/{epochs}, Generator Loss: {epoch_g_loss/n_batches:.4f}, Critic Loss: {epoch_c_loss/n_batches:.4f}"
        )

    return g_losses, c_losses

# ENHANDCED MONITORING OF TRAINING PROGRESS
# for epoch in range(epochs):
#     print(f"\nEpoch {epoch + 1}/{epochs}")
#     for i, real_data in enumerate(train_loader):
#         # Training steps here

#         if i % 10 == 0:
#             print(f"Iteration {i}: G_loss={g_loss.item():.4f}, C_loss={c_loss.item():.4f}")
#             print(f"Synthetic sample mean: {synthetic_data.mean(axis=0).mean():.4f}, std: {synthetic_data.std(axis=0).mean():.4f}")

# ### OR WHICH IS BETTER?
# print(f"Epoch: {epoch}, Batch Size: {batch_size}, Real: {real_samples.shape}, Fake: {fake_samples.shape}")
# print(f"Shape of original data: {original_data.shape}, Shape of synthetic data: {synthetic_samples.shape}")
# print(f"Critic Loss: {c_loss.item()}, Generator Loss: {g_loss.item()}")


def compute_gradient_penalty(critic, real_data, fake_data, device):
    """
    Computes the gradient penalty for WGAN-GP.

    Purpose and Importance of the Gradient Penalty:

        The gradient penalty enforces Lipschitz continuity on the Critic network by penalizing
        deviations in the gradient norm, thereby preventing the Critic from over-optimizing on real
        or fake samples. This constraint is crucial because:

    Training Stability: It keeps the model stable by preventing excessively steep gradients, which can cause instability.

    Quality of Generated Samples: The penalty helps ensure that the Generator produces samples that align
    with the distribution of real data by maintaining the Critic’s function smoothness, which guides
    the Generator more effectively.

    How compute_gradient_penalty Works
    1. Random Weighting (Interpolation):
        The function starts by creating a random weighting (alpha) between the real and generated samples.
        This weighting produces an "interpolated" sample halfway between real and fake data, which lies
        along the gradient path.
        The alpha tensor, sampled from a uniform distribution, gives each interpolated data point a unique
        position on this path.

    2. Gradient Computation:
        The function then computes the gradient of the Critic’s output with respect to this interpolated sample.
        The torch.autograd.grad function calculates these gradients, which measure how much the Critic’s score
        would change with minor changes to the interpolated input, representing the smoothness of the Critic’s
        function along the interpolated path.

    3. Gradient Penalty Calculation:
        The function calculates the gradient penalty by comparing the gradient norm to the ideal norm of 1
        (enforcing Lipschitz continuity). Any deviation from this ideal norm contributes to the penalty.
        The expression ((gradients.norm(2, dim=1) - 1) ** 2).mean() measures the squared deviation of the
        gradient norm from 1. Averaging these deviations over the batch produces a single penalty value.

    Parameters:
        critic (nn.Module): Critic model.
        real_data (torch.Tensor): Batch of real data samples.
        fake_data (torch.Tensor): Batch of generated (fake) data samples.
        device (torch.device): Device (CPU/GPU) for computation.

    Returns:
        torch.Tensor: Computed gradient penalty.
    """
    alpha = torch.rand(real_data.size(0), 1, device=device, requires_grad=True)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)
    gradients = torch.autograd.grad(
        outputs=critic(interpolated).sum(),
        inputs=interpolated,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def plot_losses(g_losses, c_losses, epoch):
    print(f"Plotting losses for Epoch {epoch}...")
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label="Generator Loss", color="blue")
    plt.plot(c_losses, label="Critic Loss", color="red")
    plt.title(f"Training Losses - Epoch {epoch}")
    plt.xlabel("Training Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{config.RESULT_DIR}/loss_plot_epoch_{epoch}.png")
    plt.close()

# New parameter for target synthetic samples count
TARGET_SYNTHETIC_COUNT = 77

def train_and_generate(scaled_data, trainable_col_names, epochs, scaler):
    """
    Trains the WGAN-GP model and generates synthetic data.

    Parameters:
        scaled_data (np.ndarray): Scaled version of the original training data.
        original_data (pd.DataFrame): Original training data.
        epochs (int): Number of training epochs.
        scaler (RobustScaler): Scaler object used for original data preprocessing.

    Returns:
        pd.DataFrame: Concatenated synthetic data generated across K-Fold splits.
        pd.DataFrame: Original data.
    """

    synthetic_samples = []
    kfold = KFold(n_splits=config.CV_N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    samples_per_fold = TARGET_SYNTHETIC_COUNT // config.CV_N_SPLITS

    for _, (train_idx, _) in enumerate(kfold.split(scaled_data)):
        train_loader = DataLoader(
            torch.tensor(scaled_data[train_idx], dtype=torch.float32),
            batch_size=config.BATCH_SIZE,
        )
        # Initialize Generator and Critic
        generator = Generator(scaled_data.shape[1], scaled_data.shape[1]).to(
            config.DEVICE
        )
        critic = Critic(scaled_data.shape[1]).to(config.DEVICE)

        g_optimizer = Adam(generator.parameters(), lr=config.LEARNING_RATE)
        c_optimizer = Adam(critic.parameters(), lr=config.LEARNING_RATE)

        # Train WGAN-GP
        train_wgan_gp(
            train_loader,
            generator,
            critic,
            g_optimizer,
            c_optimizer,
            config.DEVICE,
            epochs,
        )

        # Generate specific count of synthetic samples per fold

        # Generate synthetic data for this fold
        num_samples = 77  # Adjust the number of synthetic samples to reach your target count
        latent_samples = torch.randn(num_samples, generator.input_dim, device=config.DEVICE)
        with torch.no_grad():
            generator.eval()
            synthetic_samples = generator(latent_samples).cpu().numpy()
        
        synthetic_data = generator(latent_samples).cpu().detach().numpy()

        # Recenter and add synthetic samples
        synthetic_data = recenter_data(synthetic_data, scaled_data[train_idx], scaler)
        synthetic_samples.append(pd.DataFrame(synthetic_data, columns=trainable_col_names))

    # Concatenate all synthetic data across folds to reach target count
    return pd.concat(synthetic_samples, ignore_index=True), pd.DataFrame(scaled_data, columns=trainable_col_names)
    