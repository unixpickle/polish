# polish

This is a simple deep learning system for denoising ray traced images. It includes:

 * A program to create a denoising dataset from scratch
 * A training pipeline in [PyTorch](https://pytorch.org/)
 * A Go inference library with pre-trained models
 * A command-line utility for denoising images

Currently, all denoisers operate only on the RGB channels in an image. I am currently generating a large dataset of renders with extra features (albedo and ray collision angles). This should allow an integrated denoiser with better performance on synthetic scenes.

# Usage

**Note:** this code expects a version of Go that supports modules. Ideally, version 1.14 or later. See the [Go downloads page](https://golang.org/dl/).

## Command-line interface

To build the command-line tool, simply clone this repository (outside of your `GOPATH`) and run:

```
$ go build -o polish_cli
```

Now you can run the `polish_cli` binary to denoise an image:

```
./polish_cli input.png output.png
```

## Go API

There is also a Go API for `polish`, implemented in the [polish](polish) sub-directory. The main API is `PolishImage`:

```go
func PolishImage(t ModelType, img image.Image) image.Image
```

For example, you could use the built-in deep CNN model as follows:

```go
output := polish.PolishImage(polish.ModelTypeDeep, input)
```

# Getting data

The [create_dataset](create_dataset) directory contains a Go program that creates random scenes and renders them to produce a denoising dataset. It expects to use models from [ModelNet40](https://modelnet.cs.princeton.edu/), and textures from ImageNet (or any directory of images, really). It generates scenes by selecting a layout type (either a boxed room, or a large dome), randomizing lighting, loading and positioning various 3D models, and selecting random textures and materials for all models and walls.

Creating the dataset is rather costly, since it involves ray tracing many random and complex scenes to convergence. I will soon provide a premade dataset that can be used as the standard choice.

# Training

The [training](training) directory contains a Python program to train a denoising neural network. It processes data produced by `create_dataset`, and automatically performs data augmentation and other tricks using that data. It includes a Jupyter notebook for converting the finished PyTorch models into Go source files that can be integrated into the Go package.
