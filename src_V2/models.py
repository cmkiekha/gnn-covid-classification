import torch
from torch import nn

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
        self.input_dim = input_dim
        self.output_dim = output_dim
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
    device: torch.device = config.DEVICE,
) -> torch.Tensor:
    """
    Computes the gradient penalty for enforcing the Lipschitz constraint in WGAN-GP.
    This penalty promotes smooth gradients of the critic network.

    Args:
        critic (nn.Module): The critic network that evaluates the authenticity of data.
        real_samples (torch.Tensor): Samples from the real dataset.
        fake_samples (torch.Tensor): Generated samples from the generator.
        device (torch.device): The device tensors are on (e.g., CPU or CUDA), defaults to configured device.

    Returns:
        torch.Tensor: The computed gradient penalty, a scalar tensor that should be added
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
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty