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

# Training your own models

The built-in pre-trained models should be sufficient for most use cases. However, if you do need to train your own model, this repository includes everything needed to create a dataset and train a model on it.

## Getting data

You will likely want to get started by downloading the ~1GB [data_608.tar](https://polish.aqnichol.com/data_608.tar) dataset, which includes 608 rendered scenes. Creating this dataset took 4,600 CPU hours, which translates to roughly 20 days on an average workstation.

The dataset was created with the [create_dataset](create_dataset) program, which creates random scenes and renders them at various rays-per-pixel. It expects to use models from [ModelNet40](https://modelnet.cs.princeton.edu/), and textures from ImageNet (or any directory of images, really). It generates scenes by selecting a layout type (either a boxed room or a large dome), randomizing lighting, loading and positioning various 3D models, and selecting random textures and materials for all models and walls.

## Training with PyTorch

The [training](training) directory contains a Python program to train a denoising neural network. It processes data produced by `create_dataset`, and automatically performs data augmentation and other tricks using that data. It includes a Jupyter notebook for converting the finished PyTorch models into Go source files that can be integrated into the Go package.
