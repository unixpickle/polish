import argparse
import os

from PIL import Image
import numpy as np
import torch
import torch.optim as optim

from polish.baseline import identity_baseline
from polish.dataset import PolishDataset
from polish.models import all_models


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    args = arg_parser().parse_args()

    models = all_models(aux=args.aux)
    if args.model_type not in models:
        raise ValueError('unknown model: ' + args.model_type)
    model = models[args.model_type]
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.to(device)

    trains, tests = create_datasets(args.data, args.batch, aux=args.aux)
    print('baseline: train %f, test %f' % (identity_baseline(trains), identity_baseline(tests)))

    opt = optim.Adam(model.parameters(), lr=args.lr)

    i = 0
    for (train_in, train_out), (test_in, test_out) in zip(trains, tests):
        train_in, train_out = train_in.to(device), train_out.to(device)
        test_in, test_out = test_in.to(device), test_out.to(device)

        train_loss = model.loss(train_in, train_out)
        with torch.no_grad():
            model.eval()
            test_loss = model.loss(test_in, test_out)
            model.train()

        opt.zero_grad()
        train_loss.backward()
        opt.step()

        if not i % args.save_interval:
            torch.save(model.state_dict(), args.model_path)
            with torch.no_grad():
                model.eval()
                test_pred = model(test_in)
                model.train()
                save_rendering(test_in, test_pred)

        print('step %d: train=%f test=%f' % (i, train_loss.item(), test_loss.item()))
        i += 1


def create_datasets(data_dir, batch, **kwargs):
    dl_kwargs = {'num_workers': 8, 'pin_memory': True, 'batch_size': batch}
    train_loader = torch.utils.data.DataLoader(PolishDataset(data_dir, extra_aug=True, **kwargs),
                                               **dl_kwargs)
    test_loader = torch.utils.data.DataLoader(PolishDataset(data_dir, train=False, **kwargs),
                                              **dl_kwargs)
    return train_loader, test_loader


def save_rendering(inputs, outputs):
    joined = torch.cat([inputs[:, :3], outputs], dim=-1).permute(0, 2, 3, 1).contiguous()
    joined = joined.view(-1, *joined.shape[2:])
    arr = joined.detach().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype('uint8')
    Image.fromarray(arr).save('samples.png')


def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', default='../data')
    parser.add_argument('--aux', action='store_true')
    parser.add_argument('--model-path', default='model.pt')
    parser.add_argument('--model-type', default='shallow')
    parser.add_argument('--save-interval', default=10, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch', default=4, type=int)
    return parser


if __name__ == '__main__':
    main()
