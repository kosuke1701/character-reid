from collections import defaultdict
import glob
import json

from PIL import Image

import numpy as np

from scipy.optimize import brentq
from scipy.interpolate import interp1d

from sklearn.utils import shuffle
from sklearn.metrics import roc_curve

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

def _load_image(filename):
    try:
        with open(filename, "rb") as f:
            image = Image.open(f)
            return image.convert("RGB")
    except UserWarning as e:
        print(filename)
        input("Something wrong happens while loading image: {}".format(filename))

class ImageDataset(Dataset):
    def __init__(self, lst_img_fn, lst_label, transform=None):
        assert len(lst_img_fn) == len(lst_label)
        self.lst_img_fn = lst_img_fn
        self.lst_label = lst_label

        self.transform = transform

    def __len__(self):
        return len(self.lst_img_fn)
    
    def __getitem__(self, idx):
        img = _load_image(self.lst_img_fn[idx])
        if self.transform is not None:
            img = self.transform(img)
        
        target = self.lst_label[idx]

        return img, target

def collate_fn(data_batch):
    data = [tup[0].unsqueeze(0) for tup in data_batch]
    labels = [tup[1] for tup in data_batch]
    data = torch.cat(data)
    labels = torch.LongTensor(labels)
    return data, labels

##
## For re-identification dataset
##

def _create_index_image_map(root):
    lst_image_fn = glob.glob(f"{root}/*")

    idx_img_map = {}
    for img_fn in lst_image_fn:
        idx = img_fn.split("/")[-1].split(".")[0]
        idx = int(idx)

        idx_img_map[idx] = img_fn
    
    return idx_img_map

def create_dataset(root, data, transform):
    """
    Arguments:
        data - [i_character: [index_of_tag, list_of_illust_ids]]]
    """
    idx_img_map = _create_index_image_map(root)

    char_idx = defaultdict(list)
    lst_fn = []
    lst_label = []
    for i_char, lst_img_idx in data:
        for img_idx in lst_img_idx:
            data_idx = len(lst_fn)

            lst_fn.append(idx_img_map[img_idx])
            lst_label.append(i_char)

            char_idx[i_char].append(data_idx)
    
    dataset = ImageDataset(lst_fn, lst_label, transform)

    return dataset, char_idx

class MetricBatchSampler(BatchSampler):
    def __init__(self, dataset, char_idx, n_max_per_char, n_batch_size, n_random):
        self.char_idx = char_idx
        self.n_data = len(dataset)

        self.n_max_per_char = n_max_per_char
        self.n_batch_size = n_batch_size
        self.n_random = n_random
    
    def __len__(self):
        return 1000000000
    
    def __iter__(self):
        segments = []
        for i_char, lst_data_idx in self.char_idx.items():
            lst_data_idx = shuffle(lst_data_idx)
            for i in range(0, len(lst_data_idx), \
                self.n_max_per_char):
                new_seg = lst_data_idx[i:i+self.n_max_per_char]
                if len(new_seg) > 1:
                    segments.append(new_seg)
        segments = shuffle(segments)

        while True:
            batch = []

            # Positive data
            flag_continue = True
            while len(batch) < self.n_batch_size:
                if len(segments) == 0:
                    flag_continue = False
                    break
                batch += segments.pop()
            if not flag_continue:
                break

            # Negative data
            for i in range(self.n_random):
                batch.append(np.random.randint(self.n_data))
            
            yield batch
        
        raise StopIteration

def calculate_eer(y, y_score):
    fpr, tpr, threshold = roc_curve(y, y_score)

    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, threshold)(eer)

    return eer, thresh

if __name__=="__main__":
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid
    from torchvision import transforms

    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    def imshow(img):
        print(img.size())
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.savefig("tmp.png", dpi=300)

    with open("data/10_character_reid_dataset_191114_5split.json") as h:
        data = json.load(h)
    data = sum(data, [])

    trans = [
        transforms.Resize((224,224)),
        transforms.CenterCrop(224),
        transforms.ToTensor()#,
        # transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # )
    ]
    trans = transforms.Compose(trans)

    dataset, char_idx = create_dataset(
        "/data_ssd/pixiv/illusts",
        data, trans
    )

    sampler = MetricBatchSampler(dataset, char_idx,
        n_max_per_char=3, n_batch_size=10, n_random=5)

    dataloader = DataLoader(dataset, batch_sampler=sampler,
        collate_fn = collate_fn_multi)

    dataloader_iter = iter(dataloader)
    images, target = dataloader_iter.next()
    print(target)

    print(images.size())
    imshow(make_grid(images, nrow=3))

