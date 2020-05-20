import argparse

import torch
import torch.nn as nn

from polish.models import all_models


def main():
    args = arg_parser().parse_args()

    models = all_models()
    if args.model_type not in models:
        raise ValueError('unknown model: ' + args.model_type)
    model = models[args.model_type]
    # Get running stats going.
    model.train()
    for i in range(3):
        model(torch.randn(4, 3, 64, 64))
    model.eval()

    # Set all biases and scales high to avoid
    # hitting ReLUs.
    for p in model.parameters():
        if len(p.shape) == 1:
            p.data.detach().fill_(100.0)

    image = nn.Parameter(torch.randn(3, 512, 512))
    out = model(image[None])
    px = torch.sum(out[0, :, 256, 256])
    px.backward()

    bits = torch.abs(image.grad) > 1e-8
    bitseq = torch.sum(bits.long(), dim=(0, 1))
    print('radius: %.1f' % bitseq_radius(bitseq.numpy()))


def bitseq_radius(seq):
    min_idx = -1
    max_idx = 0
    for i, x in enumerate(seq):
        if x:
            if min_idx == -1:
                min_idx = i
            max_idx = i
    return (max_idx-min_idx) / 2


def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-type', default='shallow')
    return parser


if __name__ == '__main__':
    main()
