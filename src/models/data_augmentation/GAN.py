import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt

from src.utils.preprocessing import *

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
            nn.Tanh()
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
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    # Compute gradient penalty for WGAN-GP
    alpha = torch.rand((real_samples.size(0), 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
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

def train_wgan_gp(data_loader, generator, critic, g_optimizer, c_optimizer, device, epochs=100, critic_iterations=5, lambda_gp=10):
    generator.train()
    critic.train()

    g_losses = []
    c_losses = []

    for epoch in tqdm(range(epochs), desc="Training WGAN-GP"):
        for i, data in enumerate(data_loader):
            real_samples = data[0].to(device)
            current_batch_size = real_samples.size(0)

            # Train Critic
            for _ in range(critic_iterations):
                fake_samples = generator(torch.randn(current_batch_size, generator.input_dim, device=device))
                real_scores = critic(real_samples)
                fake_scores = critic(fake_samples)
                gradient_penalty = compute_gradient_penalty(critic, real_samples, fake_samples, device)
                c_loss = -(torch.mean(real_scores) - torch.mean(fake_scores)) + lambda_gp * gradient_penalty
                critic.zero_grad()
                c_loss.backward()
                c_optimizer.step()

            # Train Generator
            fake_samples = generator(torch.randn(current_batch_size, generator.input_dim, device=device))
            fake_scores = critic(fake_samples)
            g_loss = -torch.mean(fake_scores)
            generator.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if i % 5 == 0:
                print(f"Epoch: {epoch}, Batch: {i}, G_loss: {g_loss.item()}, C_loss: {c_loss.item()}")
                g_losses.append(g_loss.item())
                c_losses.append(c_loss.item())

    plt.figure(figsize=(6, 6))
    plt.plot(range(len(g_losses)), g_losses, marker='o')
    plt.title('Generator Loss')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.plot(range(len(c_losses)), c_losses, marker='o')
    plt.title('Critic Loss')
    plt.grid(True)
    plt.show()

def train_and_generate(filepath, batch_size=32, epochs=100, device='cpu'):

    _, tensor_data, scaled_data, _, _ = process(filepath)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = scaled_data.shape[1]

    data_loader = DataLoader(tensor_data, batch_size=batch_size, shuffle=True)

    generator = Generator(input_dim, input_dim).to(device)
    critic = Critic(input_dim).to(device)

    g_optimizer = Adam(generator.parameters())
    c_optimizer = Adam(critic.parameters())

    train_wgan_gp(data_loader, generator, critic, g_optimizer, c_optimizer, device, epochs)

    # Generate samples and compute statistical metrics
    num_samples = 77
    latent_samples = torch.randn(num_samples, generator.input_dim, device=device)
    generated_samples = generator(latent_samples).detach().cpu().numpy()
    return generated_samples
