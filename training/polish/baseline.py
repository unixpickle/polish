"""
Baseline methods for denoising images.
"""

import torch


def identity_baseline(dataset, num_samples=200):
    i = 0
    total_loss = 0
    count = 0
    for inputs, outputs in dataset:
        total_loss += torch.mean(torch.abs(inputs - outputs)).item()
        count += 1
        i += inputs.shape[0]
        if i >= num_samples:
            break
    return total_loss / count
