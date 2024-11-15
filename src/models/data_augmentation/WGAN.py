
import sys
import os

# Add the path containing the 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.utils.preprocessing import process


import src.config as config
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from sklearn.model_selection import KFold
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src.utils.preprocessing import *
from src.utils.evaluation import compare_distributions
import matplotlib.pyplot as plt
from src.utils.preprocessing import *
import src.config as config
from tqdm import tqdm


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.input_dim = input_dim
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

    def forward(self, z):
        return self.net(z)

class Critic(nn.Module):
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
        return self.net(x)

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    # Compute gradient penalty for WGAN-GP
    alpha = torch.rand((real_samples.size(0), 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True
    )
    d_interpolates = critic(interpolates)
    fake = torch.ones((real_samples.size(0), 1), requires_grad=False, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_wgan_gp(
    data_loader,
    generator,
    critic,
    g_optimizer,
    c_optimizer,
    device,
    epochs=100,
    critic_iterations=5,
    lambda_gp=10,
    clip_value=1.0,  # Gradient clipping value
):
    generator.train()
    critic.train()
    
    g_losses, c_losses = [], []

    for epoch in tqdm(range(epochs), desc="Training WGAN-GP"):
        print(f"Starting Epoch {epoch + 1}/{epochs}")  # Print start of each epoch
        for batch_idx, data in enumerate(data_loader):
            real_samples = data[0].to(device)
            current_batch_size = real_samples.size(0)

            # Critic Training
            for _ in range(critic_iterations):
                fake_samples = generator(
                    torch.randn(current_batch_size, generator.input_dim, device=device)
                )

                real_scores = critic(real_samples)
                fake_scores = critic(fake_samples)

                # Gradient penalty computation
                gradient_penalty = compute_gradient_penalty(
                    critic, real_samples, fake_samples, device
                )
                c_loss = -(torch.mean(real_scores) - torch.mean(fake_scores)) + lambda_gp * gradient_penalty

                c_optimizer.zero_grad()
                c_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), clip_value)
                c_optimizer.step()

                # NaN check for stability
                if torch.isnan(c_loss):
                    print("NaN detected in critic loss.")
                    break  # Early exit if NaNs are detected

            # Generator Training
            fake_samples = generator(
                torch.randn(current_batch_size, generator.input_dim, device=device)
            )
            fake_scores = critic(fake_samples)
            g_loss = -torch.mean(fake_scores)

            g_optimizer.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), clip_value)
            g_optimizer.step()

            # NaN check for stability
            if torch.isnan(g_loss):
                print("NaN detected in generator loss.")
                break  # Early exit if NaNs are detected

            # Print statement for current losses
            if batch_idx % 5 == 0:
                g_losses.append(g_loss.item())
                c_losses.append(c_loss.item())
                print(f"Epoch: {epoch + 1}, Batch: {batch_idx}, G_loss: {g_loss.item()}, C_loss: {c_loss.item()}")

    # Visualize training losses
    plt.figure(figsize=(12, 5))
    plt.plot(g_losses, label="Generator Loss", color="blue", alpha=0.7)
    plt.plot(c_losses, label="Critic Loss", color="red", alpha=0.7)
    plt.title("Training Losses Over Time")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


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
#     clip_value=1.0,  # Gradient clipping value
# ):
#     generator.train()
#     critic.train()
    
#     g_losses, c_losses = [], []

#     for epoch in tqdm(range(epochs), desc="Training WGAN-GP"):
#         for batch_idx, data in enumerate(data_loader):
#             real_samples = data[0].to(device)
#             current_batch_size = real_samples.size(0)

#             # Critic Training
#             for _ in range(critic_iterations):
#                 fake_samples = generator(
#                     torch.randn(current_batch_size, generator.input_dim, device=device)
#                 )

#                 real_scores = critic(real_samples)
#                 fake_scores = critic(fake_samples)

#                 # Gradient penalty computation with error handling
#                 gradient_penalty = compute_gradient_penalty(
#                     critic, real_samples, fake_samples, device
#                 )
#                 c_loss = -(torch.mean(real_scores) - torch.mean(fake_scores)) + lambda_gp * gradient_penalty

#                 c_optimizer.zero_grad()
#                 c_loss.backward()
#                 torch.nn.utils.clip_grad_norm_(critic.parameters(), clip_value)
#                 c_optimizer.step()

#                 # NaN check for stability
#                 if torch.isnan(c_loss):
#                     print("NaN detected in critic loss.")
#                     break  # Early exit if NaNs are detected

#             # Generator Training
#             fake_samples = generator(
#                 torch.randn(current_batch_size, generator.input_dim, device=device)
#             )
#             fake_scores = critic(fake_samples)
#             g_loss = -torch.mean(fake_scores)

#             g_optimizer.zero_grad()
#             g_loss.backward()
#             torch.nn.utils.clip_grad_norm_(generator.parameters(), clip_value)
#             g_optimizer.step()

#             # NaN check for stability
#             if torch.isnan(g_loss):
#                 print("NaN detected in generator loss.")
#                 break  # Early exit if NaNs are detected

#             if batch_idx % 5 == 0:
#                 g_losses.append(g_loss.item())
#                 c_losses.append(c_loss.item())
#                 print(f"Epoch: {epoch}, Batch: {batch_idx}, G_loss: {g_loss.item()}, C_loss: {c_loss.item()}")

#     # Visualize training losses
#     plt.figure(figsize=(12, 5))
#     plt.plot(g_losses, label="Generator Loss", color="blue", alpha=0.7)
#     plt.plot(c_losses, label="Critic Loss", color="red", alpha=0.7)
#     plt.title("Training Losses Over Time")
#     plt.xlabel("Iteration")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

    # plt.figure(figsize=(6, 6))
    # plt.plot(range(len(g_losses)), g_losses, marker="o")
    # plt.title("Generator Loss")
    # plt.grid(True)
    # plt.show()

    # plt.figure(figsize=(6, 6))
    # plt.plot(range(len(c_losses)), c_losses, marker="o")
    # plt.title("Critic Loss")
    # plt.grid(True)
    # plt.show()

def train_and_generate_k_fold(
    dataset,
    tensor_data,
    original_data_unscaled,
    leftout_original_data,
    scaler,
    original_dim,
    epochs,
    k=3):

    input_dim = original_data_unscaled.shape[1]
    k_fold = KFold(n_splits=k, shuffle=True, random_state=config.RANDOM_STATE)

    # Store valid results from each fold
    valid_fold_results = []

    # Add gradient clipping
    clip_value = 1.0

    for fold, (train_idx, val_idx) in enumerate(k_fold.split(tensor_data)):
        print(f"Fold {fold + 1}/{k}")

        train_subset = torch.utils.data.Subset(tensor_data, train_idx)
        train_loader = DataLoader(
            train_subset, batch_size=config.BATCH_SIZE, shuffle=True
        )

        generator = Generator(input_dim, input_dim).to(config.DEVICE)
        critic = Critic(input_dim).to(config.DEVICE)

        g_optimizer = Adam(
            generator.parameters(), 
            lr=1e-4,  # Generator learning rate (was likely 0.0002 before)
            betas=(0.0, 0.9)  # Modified betas for better stability
        )
        
        c_optimizer = Adam(
            critic.parameters(), 
            lr=2e-4,  # Critic learning rate (was likely 0.0002 before)
            betas=(0.0, 0.9)  # Modified betas for better stability
        )

        # # Initialize with proper learning rates
        # g_optimizer = Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
        # c_optimizer = Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.9))

        # Train with gradient clipping
        for epoch in range(epochs):
            generator.train()
            critic.train()

            for batch_idx, real_data in enumerate(
                train_loader
            ):
                real_data = real_data.to(config.DEVICE)
                batch_size = real_data.size(0)

                # Train Critic
                for _ in range(5):  # critic iterations
                    noise = torch.randn(batch_size, input_dim, device=config.DEVICE)
                    fake_data = generator(noise)

                    c_loss = -(
                        torch.mean(critic(real_data)) - torch.mean(critic(fake_data))
                    )

                    # Gradient penalty
                    gradient_penalty = compute_gradient_penalty(
                        critic, real_data, fake_data, config.DEVICE
                    )
                    c_loss += 10 * gradient_penalty

                    c_optimizer.zero_grad()
                    c_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), clip_value)
                    c_optimizer.step()

                # Train Generator
                noise = torch.randn(batch_size, input_dim, device=config.DEVICE)
                fake_data = generator(noise)
                g_loss = -torch.mean(critic(fake_data))

                g_optimizer.zero_grad()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), clip_value)
                g_optimizer.step()

        # Generate samples
        with torch.no_grad():
            noise = torch.randn(
                config.SAMPLE_COUNT_TO_GENERATE, input_dim, device=config.DEVICE
            )
            generated_samples = generator(noise).cpu().numpy()

            # Check for valid samples
            if not (
                np.isnan(generated_samples).any() or np.isinf(generated_samples).any()
            ):
                valid_fold_results.append(generated_samples)
    
    # Aggregate results using median instead of mean
    if len(valid_fold_results) > 0:
        all_samples = np.concatenate(valid_fold_results, axis=0)
        final_samples = np.mean(
            all_samples.reshape(len(valid_fold_results), -1, input_dim), axis=0
        )
        return pd.DataFrame(final_samples, columns=original_data_unscaled.columns)
    raise ValueError("No valid samples generated from any fold")


def train_and_generate(filepath, batch_size=32, epochs=125, device="cpu"):

    _, tensor_data, scaled_data, _, _ = process(filepath)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = scaled_data.shape[1]

    data_loader = DataLoader(tensor_data, batch_size=batch_size, shuffle=True)

    generator = Generator(input_dim, input_dim).to(device)
    critic = Critic(input_dim).to(device)

    g_optimizer = Adam(generator.parameters())
    c_optimizer = Adam(critic.parameters())

    train_wgan_gp(
        data_loader, generator, critic, g_optimizer, c_optimizer, device, epochs
    )

    # Generate samples and compute statistical metrics
    num_samples = 100
    latent_samples = torch.randn(num_samples, generator.input_dim, device=device)
    generated_samples = generator(latent_samples).detach().cpu().numpy()
    return generated_samples

