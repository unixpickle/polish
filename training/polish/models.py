"""
Machine learning models for denoising.
"""

from abc import abstractproperty

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AUX_FEATURE_CHANNELS = 4


def all_models(**kwargs):
    """
    Get a dict of supported models.
    """
    return {
        'linear': LinearDenoiser(**kwargs),
        'shallow': ShallowDenoiser(**kwargs),
        'deep': DeepDenoiser(**kwargs),
        'bilateral': BilateralDenoiser(**kwargs),
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

    def __init__(self, aux=False, kernel_size=7):
        super().__init__()
        if not kernel_size % 2:
            raise ValueError('kernel_size must be odd')
        self.conv = nn.Conv2d(3 + AUX_FEATURE_CHANNELS if aux else 3,
                              3, kernel_size, padding=kernel_size//2)

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

    def __init__(self, aux=False, kernel_size=5, hidden_size=32):
        super().__init__()
        if not kernel_size % 2:
            raise ValueError('kernel_size must be odd')
        self.conv1 = nn.Conv2d(3 + AUX_FEATURE_CHANNELS if aux else 3,
                               hidden_size, kernel_size,
                               padding=kernel_size//2)
        self.conv2 = nn.Conv2d(hidden_size, 3, kernel_size, padding=kernel_size//2)

    @property
    def dim_lcd(self):
        return 1

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x


class SepConv2d(nn.Module):
    def __init__(self, depth_in, depth_out, kernel_size, stride=1, padding=0):
        super().__init__()
        self.spatial = nn.Conv2d(depth_in, depth_in, kernel_size,
                                 stride=stride, padding=padding, groups=depth_in)
        self.depthwise = nn.Conv2d(depth_in, depth_out, 1)

    def forward(self, x):
        x = self.spatial(x)
        x = F.relu(x)
        x = self.depthwise(x)
        return x


class DeepDenoiser(Denoiser):
    """
    A denoiser that has multiple hidden layers.
    """

    def __init__(self, aux=False, conv2d=SepConv2d):
        super().__init__()
        self.conv1 = nn.Conv2d(3 + AUX_FEATURE_CHANNELS if aux else 3,
                               64, 5, padding=2, stride=2)
        self.conv2 = conv2d(64, 128, 5, padding=2, stride=2)

        self.residuals = nn.ModuleList([nn.Sequential(
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            conv2d(256, 128, 3, padding=1),
        ) for _ in range(4)])

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


class BilateralDenoiser(Denoiser):
    """
    A denoiser that uses a bilateral filter with learned
    parameters.
    """

    def __init__(self, filter_size=15, aux=False):
        super().__init__()

        if not filter_size % 2:
            raise ValueError('filter size must be odd')

        self.aux = aux
        self.filter_size = filter_size

        distances = np.zeros([filter_size]*2, dtype=np.float32)
        middle = filter_size // 2
        for i in range(filter_size):
            for j in range(filter_size):
                distances[i, j] = (i-middle)*(i-middle) + (j-middle)*(j-middle)

        self.register_buffer('distances', torch.from_numpy(distances).view(-1))
        self.blur_sigma = nn.Parameter(torch.Tensor([5.0])[0])
        self.diff_sigma = nn.Parameter(torch.Tensor([1.0])[0])

    @property
    def dim_lcd(self):
        return 1

    def forward(self, x):
        if self.aux:
            x = x[:, :3]

        # Create patches tensor: [N x C x K^2 x H x W]
        padding = self.filter_size // 2
        # Pad with huge negative values instead of zeros
        # so that the filter does not include the padding.
        padded = F.pad(x + 100, [padding] * 4) - 100
        patches = F.unfold(padded, self.filter_size)
        patches = patches.view(*x.shape[:2], self.filter_size**2, *x.shape[2:])

        diffs = torch.pow(patches - x[:, :, None], 2)
        diffs = diffs / torch.pow(self.diff_sigma, 2)

        blurs = torch.zeros_like(patches) + self.distances[None, None, :, None, None]
        blurs = blurs / torch.pow(self.blur_sigma, 2)

        probs = torch.exp(-(diffs + blurs))
        probs = probs / torch.sum(probs, dim=2, keepdim=True)

        return torch.sum(patches * probs, dim=2)
