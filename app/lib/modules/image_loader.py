from PIL import Image

import numpy as np

import torch
from torchvision import transforms

def _load_image(filename):
    try:
        with open(filename, "rb") as f:
            image = Image.open(f)
            return image.convert("RGB")
    except UserWarning as e:
        print(filename)
        input("Something wrong happens while loading image: {} {}".format(filename, str(e)))

class AbstractImageLoader(object):
    def load_images(self, lst_filenames):
        raise NotImplementedError()

class FullImageLoader(AbstractImageLoader):
    def __init__(self, transform):
        self.transform = transform
    
    def load_thumbnail(self, lst_filenames, size, callback=None):
        imgs = []
        for i_fn, fn in enumerate(lst_filenames):
            img = Image.open(fn)
            img = img.convert("RGBA")

            W,H = img.size
            if W > H:
                _img = Image.new(img.mode, (W, W))
                _img.paste(img, (0, (W-H)//2))
            elif H > W:
                _img = Image.new(img.mode, (H, H))
                _img.paste(img, ((H-W)//2, 0))
            else:
                _img = img
            _img = _img.resize((size, size))
            
            imgs.append(_img)

            if callback is not None:
                callback(i_fn, len(lst_filenames))
        return imgs

    def load_images(self, lst_filenames):
        imgs = [_load_image(fn) for fn in lst_filenames]
        if self.transform is not None:
            imgs = [self.transform(img) for img in imgs]
        imgs = torch.cat([img.unsqueeze(0) for img in imgs])

        return imgs
