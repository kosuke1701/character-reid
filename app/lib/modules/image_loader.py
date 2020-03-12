from PIL import Image

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

    def load_images(self, lst_filenames):
        imgs = [_load_image(fn) for fn in lst_filenames]
        if self.transform is not None:
            imgs = [self.transform(img) for img in imgs]
        imgs = torch.cat([img.unsqueeze(0) for img in imgs])

        return imgs
