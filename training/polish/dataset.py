import os
import random

from PIL import Image
import numpy as np
import torch
import torch.utils.data as data


class PolishDataset(data.IterableDataset):
    def __init__(self, data_dir, split='train', crop_size=192, cache=True):
        self.cache = cache
        self.crop_size = crop_size
        self.listing = [os.path.join(data_dir, split, x)
                        for x in os.listdir(os.path.join(data_dir, split))
                        if x.endswith('.png')]

    def __iter__(self):
        cache = None
        if self.cache:
            cache = {p: np.array(Image.open(p)) for p in self.listing}
        paths = self.listing.copy()
        while True:
            random.shuffle(paths)
            for path in paths:
                yield self.get_sample(path, cache)

    def get_sample(self, img_path, cache):
        idx = random.randrange(7)
        if cache is None:
            img = np.array(Image.open(img_path))
        else:
            img = cache[img_path]
        img = img.astype('float32') / 255.0
        img_size = img.shape[1] // 2
        cx = random.randrange(img_size - self.crop_size)
        cy = random.randrange(img_size - self.crop_size)
        f = random.random() < 0.5
        return (self.get_subimage(img, idx % 7, cx, cy, f),
                self.get_subimage(img, 7, cx, cy, f))

    def get_subimage(self, img, idx, crop_x, crop_y, flip):
        sub_img_size = img.shape[1] // 2
        sub_idx = idx % 7
        col = (sub_idx % 2) * sub_img_size
        row = (sub_idx // 2) * sub_img_size
        result = img[row:row+sub_img_size, col:col+sub_img_size]
        result = result[crop_x:, crop_y:][:self.crop_size, :self.crop_size]
        if flip:
            result = np.array(result[:, ::-1])
        return torch.from_numpy(result).permute(2, 0, 1).contiguous()
