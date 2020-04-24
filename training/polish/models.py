"""
Machine learning models for denoising.
"""

from abc import abstractmethod, abstractproperty
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def all_models():
    """
    Get a dict of supported models.
    """
    return {
        'linear': LinearDenoiser(),
        'shallow': ShallowDenoiser(),
        'deep': DeepDenoiser(),
    }


class Denoiser(nn.Module):
    @abstractmethod
    def forward(self, images, errors):
        """
        Attempt to denoise the images.

        Args:
            images: an [N x C x H x W] Tensor.
            errors: an [N] Tensor of approximate mean L1
                errors for each image. This may guide how
                much the model affects each image.
        """
        pass

    def loss(self, images, targets):
        """
        Compute the reconstruction loss using a noisy but
        unbiased estimate of the target errors.
        """
        errors = self.noisy_errors(images, targets)
        return torch.mean(torch.abs(self(images, errors) - targets))

    def noisy_errors(self, images, targets):
        errors = torch.mean(torch.abs(images - targets), dim=(1, 2, 3))
        errors = errors.detach()
        return errors + torch.randn_like(errors) * 0.05

    @abstractproperty
    def dim_lcd(self):
        """
        Get a factor that must divide both the width and
        height of input images.
        """
        pass


class LinearDenoiser(Denoiser):
    """
    This is the simplest possible denoiser, consisting of
    one convolutional filter.
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        if not kernel_size % 2:
            raise ValueError('kernel_size must be odd')
        self.conv = nn.Conv2d(3, 3, kernel_size, padding=kernel_size//2)

    @property
    def dim_lcd(self):
        return 1

    def forward(self, x, errors):
        return self.conv(x)


class ShallowDenoiser(Denoiser):
    """
    A denoiser that has one hidden layer and doesn't
    require any spatial LCD.
    """

    def __init__(self, kernel_size=5, hidden_size=32):
        super().__init__()
        if not kernel_size % 2:
            raise ValueError('kernel_size must be odd')
        self.conv1 = nn.Conv2d(3, hidden_size, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(hidden_size, 3, kernel_size, padding=kernel_size//2)

    @property
    def dim_lcd(self):
        return 1

    def forward(self, x, errors):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x


class DeepDenoiser(Denoiser):
    """
    A denoiser that has multiple hidden layers.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 4, padding=1, stride=2)
        self.deconv2 = nn.ConvTranspose2d(32, 3, 4, padding=1, stride=2)

    @property
    def dim_lcd(self):
        return 4

    def forward(self, x, errors):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = x + embed_errors(errors, x.shape[1])
        x = self.conv3(x)
        x = F.relu(x)

        x = self.deconv1(x)
        x = F.relu(x)
        x = self.deconv2(x)
        return x


def embed_errors(errors, dim):
    expanded = errors[:, None].repeat(1, dim//2)
    phases = torch.Tensor([i * math.pi * 2 for i in range(1, dim//2+1)])
    phases = phases.to(errors.device).to(errors.dtype)
    args = expanded * phases
    results = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    return results[..., None, None]
