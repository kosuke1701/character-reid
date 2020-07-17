from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
import glob
import json
import logging
import sys
import time

from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000

import numpy as np

from scipy.optimize import brentq
from scipy.interpolate import interp1d

from sklearn.utils import shuffle
from sklearn.metrics import roc_curve

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms
from torchvision.transforms.functional import crop, resize, pad

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
    except Exception as e:
        print(filename)
        raise(e)

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

class ImageBBDataset(ImageDataset):
    def __init__(self, lst_img_fn, lst_label, lst_bbox, bbox_scale=1.5,
        post_crop_transform=None, collate_fn_bbox=None):
        super().__init__(lst_img_fn, lst_label)

        self.lst_bbox = lst_bbox # idx_img -> list of [x_min, y_min, x_max, y_max]
        self.scale = bbox_scale

        self.post_transform = post_crop_transform
        self.bbox_collate_fn = collate_fn_bbox

    def __getitem__(self, idx):
        # get PIL image and label
        img, target = super().__getitem__(idx)

        # Crop each bounding boxes
        crop_imgs = []
        bboxs = self.lst_bbox[idx]
        assert len(bboxs) > 0
        for bbox in bboxs:
            xmin, ymin, xmax, ymax = self.rescale_bbox(*bbox)
            crop_img = crop(img, ymin, xmin, ymax-ymin, xmax-xmin)
            if self.post_transform is not None:
                crop_img = self.post_transform(crop_img)
            crop_imgs.append(crop_img)
        
        if self.bbox_collate_fn is not None:
            crop_imgs = self.bbox_collate_fn(crop_imgs)

        return crop_imgs, target
    
    def rescale_bbox(self, xmin, ymin, xmax, ymax):
        xcenter = (xmin + xmax) / 2
        xwid = xmax - xmin
        ycenter = (ymin + ymax) / 2
        ywid = ymax - ymin

        new_bbox = [
            xcenter - xwid/2*self.scale,
            ycenter - ywid/2*self.scale,
            xcenter + xwid/2*self.scale,
            ycenter + ywid/2*self.scale
        ]
        return list(map(int, new_bbox))

def bbox_collate_fn(bbox_imgs, max_bb_num=5):
    if len(bbox_imgs) > max_bb_num:
        lst_idx = np.random.choice(len(bbox_imgs), size=max_bb_num, 
            replace=False)
        bbox_imgs = [bbox_imgs[_] for _ in lst_idx]

    # 1 x C x H x W
    imgs = [img.unsqueeze(0) for img in bbox_imgs]
    masks = [1.0]*len(imgs)
    while len(imgs) < max_bb_num:
        imgs.append(torch.zeros_like(imgs[0]))
        masks.append(0.0)
    imgs = torch.cat(imgs, dim=0)
    masks = torch.FloatTensor(masks)

    return imgs, masks # (F x C x H x W), (F,)

class PadResize(object):
    def __init__(self, size, fill=0, padding_mode="constant", interpolation=2):
        self.fill = fill
        self.padding_mode = padding_mode

        self.size = size
        self.interpolation = interpolation
    
    def __call__(self, img):
        width, height = img.size
        max_size = max(width, height)

        left, right, top, bottom = 0, 0, 0, 0
        if width > height:
            top = bottom = (width - height) // 2
            if (width - height) % 2 == 1:
                top += 1
        elif height > width:
            left = right = (height - width) // 2
            if (height - width) % 2 == 1:
                left += 1
        
        img = pad(img, (left, top, right, bottom), fill=self.fill,
            padding_mode=self.padding_mode)
        img = resize(img, size=self.size, interpolation=self.interpolation)

        return img

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
    for i_char, (_, lst_img_idx) in enumerate(data):
        for img_idx in lst_img_idx:
            data_idx = len(lst_fn)

            lst_fn.append(idx_img_map[img_idx])
            lst_label.append(i_char)

            char_idx[i_char].append(data_idx)

    dataset = ImageDataset(lst_fn, lst_label, transform)

    return dataset, char_idx

def create_datasetBB(root, data, positions, **kwargs):
    """
    Note:
        **kwargs are arguments of ImageBBDataset

    Args:
        root (str): directory which contains images.
        data (list): each element is [character_ID, list_of_image_IDs]
        positions (dict): image_ID -> list of bounding boxes.
    """
    idx_img_map = _create_index_image_map(root)

    char_idx = defaultdict(list)
    lst_fn = []
    lst_label = []
    lst_bbox = []
    n_not_found = 0
    for i_char, (_, lst_img_idx) in enumerate(data):
        for img_idx in lst_img_idx:
            data_idx = len(lst_fn)

            if img_idx not in idx_img_map:
                n_not_found += 1
                continue

            lst_fn.append(idx_img_map[img_idx])
            lst_label.append(i_char)
            lst_bbox.append(positions[img_idx])

            char_idx[i_char].append(data_idx)
    
    logger.warning(f"Couldn't found {n_not_found} images during creating dataset.")
    
    dataset = ImageBBDataset(lst_fn, lst_label, lst_bbox, **kwargs)

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
    # Split characters in evaluation data into multiple set.
    _time_start_dataloaders = time.time()

    dataloaders = []
    for i in range(n_split):
        dataset, dev_char_idx = \
            create_dataset(args.root, data[i::n_split], trans)
        dataloader = DataLoader(
            dataset, batch_size=100,
            collate_fn = collate_fn, shuffle=False,
            num_workers=5
        )
        dataloaders.append(dataloader)

    _time_end_dataloaders = time.time()
    logger.info("Preparing dataloaders took {:.2f} seconds.".format(
        _time_end_dataloaders - _time_start_dataloaders
    ))

    return dataloaders

def prepare_evaluation_dataloadersBB(args, n_split, data, positions, **kwargs):
    # Split characters in evaluation data into multiple set.
    _time_start_dataloaders = time.time()

    dataloaders = []
    for i in range(n_split):
        dataset, dev_char_idx = \
            create_datasetBB(args.root, data[i::n_split],
                positions, **kwargs)
        dataloader = DataLoader(
            dataset, batch_size=100, shuffle=False
        )
        dataloaders.append(dataloader)

    _time_end_dataloaders = time.time()
    logger.info("Preparing dataloaders took {:.2f} seconds.".format(
        _time_end_dataloaders - _time_start_dataloaders
    ))

    return dataloaders

def default_sim_func(tensor1, tensor2):
    sim = - torch.norm(tensor1.unsqueeze(1) - tensor2.unsqueeze(0), dim=-1)
    return sim

def evaluate(args, trunk, embedder, dataloaders, sim_func=None):
    """
    Note:
        dataloader is assumed to output a tuple of (batch_img, batch_label).
        batch_img can be either of Tensor or Sequence of Tensors.
        batch_img will be input to trunk model.

    Args:
        args (Namespace): If args.normalize is True, embedding tensor
            which is returned by embedder is normalized before computing
            similarities. Normalization is applied to the last dimension.
        sim_func (function): A function which returns similarities between
            vectors in given two embeddings.
    """
    

    if sim_func is None:
        sim_func = default_sim_func

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

                if isinstance(batch_img, Sequence):
                    batch_img = [_.to(device) for _ in batch_img]
                else:
                    batch_img = batch_img.to(device)
                embs = embedder(trunk(batch_img))

                if args.normalize:
                    embs = embs / embs.norm(dim=-1, keepdim=True)

                embeddings.append(embs)
                labels += batch_label.tolist()

            print("Computing similarity scores...")
            results = []
            for i in range(len(embeddings)):
                sys.stdout.write("{}/{}\r".format(i, len(embeddings)))
                sys.stdout.flush()

                tmp = []
                for j in range(len(embeddings)):
                    sim = sim_func(embeddings[i], embeddings[j])
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
        plt.savefig("tmp.pdf", dpi=300)

    with open("data/10_character_reid_dataset_191114_5split.json") as h:
        data = json.load(h)
    data = sum(data, [])

    positions = {}
    with open("data/10_character_reid_dataset_191114_5split.json.faceBB_") as h:
        _positions = json.load(h)
    for _pos in _positions:
        positions.update(_pos)
    positions = {int(key): val for key, val in positions.items()}
    
    for data_ind in range(len(data))[::-1]:
        _, lst_i_img = data[data_ind]
        for ind in range(len(lst_i_img))[::-1]:
            if lst_i_img[ind] not in positions:
                del lst_i_img[ind]
        if len(lst_i_img) == 0:
            del data[data_ind]
    print(len(data))

    ##
    # trans = [
    #     transforms.Resize((224,224)),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor()#,
    #     # transforms.Normalize(
    #     #     mean=[0.485, 0.456, 0.406],
    #     #     std=[0.229, 0.224, 0.225]
    #     # )
    # ]
    # trans = transforms.Compose(trans)

    # dataset, char_idx = create_dataset(
    #     "/data_ssd/pixiv/illusts",
    #     data, trans
    # )

    post_crop_transform = PadResize(224)
    
    dataset, char_idx = create_datasetBB(
        "/data_ssd/pixiv/raw_illusts",
        data, positions, 
        post_crop_transform=post_crop_transform, 
        collate_fn_bbox=bbox_collate_fn,
        bbox_scale = 1.5
    )

    sampler = MetricBatchSampler(dataset, char_idx,
        n_max_per_char=3, n_batch_size=10, n_random=5)

    dataloader = DataLoader(dataset, batch_sampler=sampler)

    dataloader_iter = iter(dataloader)
    (images, masks), target = dataloader_iter.next()
    print("Image size:", images.size())
    print("Mask size:", masks.size())
    print("Label size:", target.size())

    images = images.view(-1, *(images.size()[2:]))
    imshow(make_grid(images, nrow=5))
