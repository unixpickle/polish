import os
import random

from PIL import Image
import numpy as np
import torch
import torch.utils.data as data


class PolishDataset(data.IterableDataset):
    def __init__(self, data_dir, train=True, crop_size=192, samples=(128, 512)):
        self.crop_size = crop_size
        self.samples = samples
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
        aug = Augmentation(inputs.shape[0], self.crop_size)
        return aug(inputs), aug(outputs)


class Augmentation:
    """
    An augmentation consistently augments input and output
    samples.
    """

    def __init__(self, img_size, crop_size):
        self.img_size = img_size
        self.crop_size = crop_size
        self.x = random.randrange(img_size - crop_size)
        self.y = random.randrange(img_size - crop_size)
        self.flip = random.random() < 0.5
        self.mask = (np.random.uniform(size=(img_size, img_size, 1)) > 0.5).astype('float32')

    def __call__(self, x):
        x = x.astype('float32') / 255.0
        if x.shape[1] > x.shape[0]:
            # Mix up the two samples to generate an
            # almost infinite amount of unbiased training
            # data.
            x = x[:, :self.img_size]*self.mask + x[:, self.img_size:]*(1-self.mask)
        x = x[self.y:, self.x:][:self.crop_size, :self.crop_size]
        if self.flip:
            x = x[:, ::-1]
        return torch.from_numpy(np.array(x)).permute(2, 0, 1).contiguous()
