import os
import random

from PIL import Image
import numpy as np
import torch
import torch.utils.data as data


class PolishDataset(data.IterableDataset):
    def __init__(self, data_dir, train=True, aux=False, crop_size=192, samples=(128, 512),
                 extra_aug=False):
        self.crop_size = crop_size
        self.samples = samples
        self.aux = aux
        self.extra_aug = extra_aug
        all_dirs = [x for x in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir, x)) and not x.startswith('.')]
        test_prefixes = 'abcd'
        if train:
            dirs = [x for x in all_dirs if not any(x.startswith(c) for c in test_prefixes)]
        else:
            dirs = [x for x in all_dirs if any(x.startswith(c) for c in test_prefixes)]
        self.dirs = [os.path.join(data_dir, x) for x in dirs]
        if not len(self.dirs):
            raise RuntimeError('missing data: %s (%s)' % (data_dir, str(train)))

    def __iter__(self):
        paths = self.dirs.copy()
        while True:
            random.shuffle(paths)
            for path in paths:
                sample = self.get_sample(path)
                yield sample

    def get_sample(self, path):
        input_case = random.choice(self.samples)
        outputs = np.array(Image.open(os.path.join(path, 'target.png')))
        inputs = np.array(Image.open(os.path.join(path, 'input_%d.png' % input_case)))
        aug = Augmentation(inputs.shape[0], self.crop_size, self.extra_aug)
        inputs = aug(inputs)
        outputs = aug(outputs)
        if self.aux:
            incident = np.array(Image.open(os.path.join(path, 'incidence.png')))[..., None]
            albedo = np.array(Image.open(os.path.join(path, 'albedo.png')))
            inputs = torch.cat([inputs, aug(albedo), aug(incident)], dim=0)
        return inputs, outputs


class Augmentation:
    """
    An augmentation consistently augments input and output
    samples.
    """

    def __init__(self, img_size, crop_size, extra_aug):
        self.img_size = img_size
        self.crop_size = crop_size
        self.x = random.randrange(img_size - crop_size)
        self.y = random.randrange(img_size - crop_size)
        self.flip_x = random.random() < 0.5
        self.channel_perm = [0, 1, 2]
        if extra_aug:
            self.flip_y = random.random() < 0.5
            self.rotation = random.randrange(4)
            random.shuffle(self.channel_perm)
        else:
            self.flip_y = False
            self.rotation = 0
        self.mask = (np.random.uniform(size=(img_size, img_size, 1)) > 0.5).astype('float32')

    def __call__(self, x):
        x = x.astype('float32') / 255.0
        if x.shape[1] > x.shape[0]:
            # Mix up the two samples to generate an
            # almost infinite amount of unbiased training
            # data.
            x = x[:, :self.img_size]*self.mask + x[:, self.img_size:]*(1-self.mask)
        x = x[self.y:, self.x:][:self.crop_size, :self.crop_size]
        for i in range(self.rotation):
            x = np.transpose(x, axes=[1, 0, 2])
            x = x[::-1]
        if self.flip_x:
            x = x[:, ::-1]
        if self.flip_y:
            x = x[::-1]

        if x.shape[2] == 3:
            x_copy = np.array(x)
            for i, p in enumerate(self.channel_perm):
                x[..., i] = x_copy[..., self.channel_perm[i]]

        return torch.from_numpy(np.array(x)).permute(2, 0, 1).contiguous()
