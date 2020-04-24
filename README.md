# polish

This is a simple deep learning system for denoising ray traced images. It includes:

 * A program to create a denoising dataset from scratch
 * A training pipeline in [PyTorch](https://pytorch.org/)

# Getting data

The [create_dataset](create_dataset) directory contains a Go program that creates random scenes and renders them to produce a denoising dataset. It expects to use models from [ModelNet40](https://modelnet.cs.princeton.edu/), and textures from ImageNet (or any directory of images, really). It generates scenes by selecting a layout type (either a boxed room, or a large dome), randomizing lighting, loading and positioning various 3D models, and selecting random textures and materials for all models and walls.

Creating the dataset is rather costly, since it involves ray tracing many random and complex scenes to convergence. I will soon provide a premade dataset that can be used as the standard choice.

# Training

The [training](training) directory contains a Python program to train a denoising neural network. It processes data produced by `create_dataset`, and automatically performs data augmentation and other tricks using that data.
