"""
Baseline methods for denoising images.
"""

import torch


def identity_baseline(dataset, num_iters=100):
    i = 0
    total_loss = 0
    for inputs, outputs in dataset:
        total_loss += torch.mean(torch.abs(inputs - outputs)).item()
        i += 1
        if i == num_iters:
            break
    return total_loss / num_iters
