import argparse
import os

import torch
import torch.optim as optim

from polish.dataset import PolishDataset
from polish.models import all_models


def main():
    args = arg_parser().parse_args()
    models = all_models()
    if args.model_type not in models:
        raise ValueError('unknown model: ' + args.model_type)
    model = models[args.model_type]
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))

    trains, tests = create_datasets(args.data, args.batch)

    opt = optim.Adam(model.parameters(), lr=args.lr)

    i = 0
    for (train_in, train_out), (test_in, test_out) in zip(trains, tests):
        train_loss = torch.mean(torch.abs(model(train_in) - train_out))

        with torch.no_grad():
            test_loss = torch.mean(torch.abs(model(test_in) - test_out))

        opt.zero_grad()
        train_loss.backward()
        opt.step()

        if not i % args.save_interval:
            torch.save(model.state_dict(), args.model_path)

        print('step %d: train=%f test=%f' % (i, train_loss.item(), test_loss.item()))
        i += 1


def create_datasets(data_dir, batch):
    kwargs = {'num_workers': 0, 'pin_memory': True, 'batch_size': batch}
    train_loader = torch.utils.data.DataLoader(PolishDataset(data_dir, split='train'), **kwargs)
    test_loader = torch.utils.data.DataLoader(PolishDataset(data_dir, split='test'), **kwargs)
    return train_loader, test_loader


def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', default='../data')
    parser.add_argument('--model-path', default='model.pt')
    parser.add_argument('--model-type', default='shallow')
    parser.add_argument('--save-interval', default=10, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch', default=4, type=int)
    return parser


if __name__ == '__main__':
    main()
