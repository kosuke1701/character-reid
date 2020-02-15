from collections import defaultdict
from copy import deepcopy
import glob
import json
import logging
import sys
import time

from PIL import Image

import numpy as np

from scipy.optimize import brentq
from scipy.interpolate import interp1d

from sklearn.utils import shuffle
from sklearn.metrics import roc_curve

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler

#
# Dataset
#

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


class PairDataset(Dataset):
    def __init__(self, dataset, lst_pairs, pair_labels):
        self.dataset = dataset
        self.pairs = lst_pairs
        self.labels = pair_labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx1, idx2 = self.pairs[idx]
        label = self.labels[idx]

        data1, _ = self.dataset[idx1]
        data2, _ = self.dataset[idx2]

        return data1, data2, label

    @classmethod
    def collate_fn(cls, data_batch):
        data1 = torch.cat([tup[0].unsqueeze(0) for tup in data_batch])
        data2 = torch.cat([tup[1].unsqueeze(0) for tup in data_batch])
        labels = torch.LongTensor([tup[2] for tup in data_batch])

        return data1, data2, labels

##
## For re-identification dataset
##

def remove_duplicate_images(data):
    data = deepcopy(data)

    img_chars = defaultdict(list)
    for char_pos, (i_char, lst_i_img) in enumerate(data):
        for i_img in lst_i_img:
            img_chars[i_img].append(char_pos)

    for i_img, lst_pos in img_chars.items():
        if len(lst_pos) > 1:
            for pos in lst_pos:
                data[pos][1].remove(i_img)

    return data

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

#
# Sampler
#

class MetricBatchSampler(BatchSampler):
    u"""
    MetricBatchSampler samples a set of images from sampled few labels.
    Single batch contains at most n_max_per_char images for a single label.
    The sampler samples images until batch size exceeds n_batch_size.
    Also, it randomly appends n_random images from a dataset to each batch.
    """
    def __init__(self, dataset, char_idx, n_max_per_char, n_batch_size, n_random):
        self.char_idx = char_idx
        self.n_data = len(dataset)

        self.n_max_per_char = n_max_per_char
        self.n_batch_size = n_batch_size
        self.n_random = n_random

        self.batches = []

    def create_batches(self):
        segments = []
        for i_char, lst_data_idx in self.char_idx.items():
            lst_data_idx = shuffle(lst_data_idx)
            for i in range(0, len(lst_data_idx), \
                self.n_max_per_char):
                new_seg = lst_data_idx[i:i+self.n_max_per_char]
                if len(new_seg) > 1:
                    segments.append(new_seg)
        segments = shuffle(segments)

        self.batches = []
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

            self.batches.append(batch)

    def __len__(self):
        if len(self.batches) == 0:
            self.create_batches()
        return len(self.batches)

    def __iter__(self):
        if len(self.batches) == 0:
            self.create_batches()

        for i_batch, batch in enumerate(self.batches):
            #print(i_batch)
            if i_batch == len(self.batches) - 1:
                self.batches = []
            yield batch

#
# Evaluation utilities.
#

def calculate_eer(y, y_score):
    fpr, tpr, threshold = roc_curve(y, y_score)

    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, threshold)(eer)

    return eer, (fpr, tpr, threshold, thresh)

def prepare_evaluation_dataloaders(args, n_split, data, trans):
    logger = logging.getLogger("main")

    # Split characters in evaluation data into multiple set.
    _time_start_dataloaders = time.time()

    dataloaders = []
    for i in range(n_split):
        dataset, dev_char_idx = \
            create_dataset(args.root, data[i::n_split], trans)
        dataloader = DataLoader(
            dataset, batch_size=100,
            collate_fn = collate_fn, shuffle=False
        )
        dataloaders.append(dataloader)

    _time_end_dataloaders = time.time()
    logger.info("Preparing dataloaders took {:.2f} seconds.".format(
        _time_end_dataloaders - _time_start_dataloaders
    ))

    return dataloaders

def evaluate(args, trunk, embedder, dataloaders):
    logger = logging.getLogger("main")

    device = next(trunk.parameters()).device

    trunk.eval()
    embedder.eval()

    lst_eer = []
    n_loader = len(dataloaders)
    for i_loader, dataloader in enumerate(dataloaders):
        logger.info(f"Dataloader {i_loader}/{n_loader}")
        _time_start_eval = time.time()

        with torch.no_grad():
            embeddings = []
            labels = []

            print("Computing embeddings...")
            n_batch = len(dataloader)
            for i_batch, (batch_img, batch_label) in enumerate(dataloader):
                sys.stdout.write(f"{i_batch}/{n_batch}\r")
                sys.stdout.flush()

                batch_img = batch_img.to(device)
                embs = embedder(trunk(batch_img))

                if args.normalize:
                    embs = embs / embs.norm(dim=1, keepdim=True)

                embeddings.append(embs)
                labels += batch_label.tolist()

            print("Computing similarity scores...")
            results = []
            for i in range(len(embeddings)):
                sys.stdout.write("{}/{}\r".format(i, len(embeddings)))
                sys.stdout.flush()

                tmp = []
                for j in range(len(embeddings)):
                    sim = torch.sum(
                        embeddings[i].unsqueeze(1) * \
                        embeddings[j].unsqueeze(0),
                        dim = 2
                    )
                    tmp.append(sim)

                row = np.array(torch.cat(tmp, dim=1).tolist(), dtype=np.float32)
                results.append(row)
        results = np.concatenate(results, axis=0)

        labels = np.array(labels, dtype=np.int8)
        labels = np.equal(labels[:,np.newaxis], labels[np.newaxis,:])
        labels = labels.astype(dtype=np.int8)

        print("Computing EER...")
        eer, _ = calculate_eer(labels.flatten(), results.flatten())
        lst_eer.append(eer)
        logger.info(f"Dataloader EER:\t{eer}")

        _time_end_eval = time.time()
        logger.info("Computing EER took {:.2f} seconds".format(
            _time_end_eval - _time_start_eval
        ))

    return np.mean(lst_eer), np.std(lst_eer)

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
