import argparse

from PIL import Image
import numpy as np
import torch

from polish.models import all_models


def main():
    args = arg_parser().parse_args()

    models = all_models()
    if args.model_type not in models:
        raise ValueError('unknown model: ' + args.model_type)
    model = models[args.model_type]
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()

    img_in = torch.from_numpy(np.array(Image.open(args.image_in))).float() / 255.0
    img_in = img_in.permute(2, 0, 1)
    with torch.no_grad():
        img_out = model(img_in[None], torch.Tensor([0.0]))
    img_out = img_out[0].permute(1, 2, 0)
    img_out = img_out.clamp(0, 1) * 255
    pil_img = Image.fromarray(img_out.detach().cpu().numpy().astype('uint8'))
    pil_img.save(args.image_out)


def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-path', default='model.pt')
    parser.add_argument('--model-type', default='shallow')
    parser.add_argument('image_in')
    parser.add_argument('image_out')
    return parser


if __name__ == '__main__':
    main()
