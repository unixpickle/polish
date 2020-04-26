"""
Machine learning models for denoising.
"""

from abc import abstractproperty

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
    def loss(self, images, targets):
        """
        Compute the reconstruction loss using a noisy but
        unbiased estimate of the target errors.
        """
        return torch.mean(torch.abs(self(images) - targets))

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

    def forward(self, x):
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

    def forward(self, x):
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
        self.conv1 = nn.Conv2d(3, 64, 5, padding=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 5, padding=2, stride=2)

        self.residuals = nn.ModuleList([nn.Sequential(
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
        ) for _ in range(3)])

        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, padding=1, stride=2)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, padding=1, stride=2)
        self.conv3 = nn.Conv2d(32, 3, 3, padding=1)

    @property
    def dim_lcd(self):
        return 4

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)

        for r in self.residuals:
            x = x + r(x)

        x = self.deconv1(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        return x
