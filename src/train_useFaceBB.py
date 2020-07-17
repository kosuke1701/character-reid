import sys

import logging


import argparse
import datetime
from functools import partial
import json
import time

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from radam import RAdam

from pytorch_metric_learning.utils.loss_and_miner_utils import get_all_triplets_indices

from tqdm import tqdm

from util import create_datasetBB, MetricBatchSampler, \
    prepare_evaluation_dataloadersBB, evaluate, remove_duplicate_images, \
    PadResize, bbox_collate_fn, default_sim_func
from model import Identity, Normalize

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Masked Embedding
class Embedding(object):
    def __init__(self, tensor, mask):
        self.tensor = tensor
        self.mask = mask

    def __getitem__(self, idx):
        return Embedding(self.tensor[idx], self.mask[idx])
    
    def to(self, device):
        return Embedding(self.tensor.to(device), self.mask.to(device))

class BoundingBoxTrunkModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.trunk = models.resnet18(pretrained=True)
        self.output_size = self.trunk.fc.in_features
        self.trunk.fc = Identity()
    
    def forward(self, x):
        """
        Args:
            x (Sequence[FloatTensor, FloatTensor]): A sequence of two Tensors.

                First Tensor has size (N x F x C x H x W), where
                N is batch dimension, F is bounding box dimension, and
                C,H,W are image dimension with 3 channels.

                Second Tensor has size (N x F). It represents whether each bounding box
                is valid or not.
        """
        img, mask = x
        n_batch, n_bb = img.size()[:2]
        img = img.view(-1, *(img.size()[2:]))

        embs = self.trunk(img)
        embs = embs.view(n_batch, n_bb, *(embs.size()[1:]))

        return Embedding(embs, mask)

class BoundingBoxEmbedder(nn.Module):
    def __init__(self, module):
        super().__init__()

        self.module = module
    
    def forward(self, x):
        embs, mask = x.tensor, x.mask

        n_batch, n_bb = embs.size()[:2]
        embs = embs.view(-1, *(embs.size()[2:]))
        x = self.module(embs)
        x = x.view(n_batch, n_bb, *(x.size()[1:]))

        return Embedding(x, mask)

def sim_func(emb1, emb2):
    tensor1, mask1 = emb1.tensor, emb1.mask
    tensor2, mask2 = emb2.tensor, emb2.mask

    # tensor1: (N x F x D)
    # tensor2: (N x F' x D)
    # sim: (N x F x F' x D)
    sim = - torch.norm(
        tensor1.unsqueeze(2) - \
            tensor2.unsqueeze(1), dim=-1
    )

    # mask1: (N x F)
    # mask2: (N x F')
    # mask: (N x F x F')
    mask = mask1.unsqueeze(2) * \
        mask2.unsqueeze(1)

    sim = sim.masked_fill(mask < 0.5, float("-inf"))
    sim = sim.view(sim.size(0), -1)

    return torch.max(sim, dim=-1)[0]

# For evaluation
def sim_func_pair(emb1, emb2):
    tensor1, mask1 = emb1.tensor, emb1.mask
    tensor2, mask2 = emb2.tensor, emb2.mask

    s1, s2 = tensor1.size(0), tensor2.size(0)
    tensor1 = tensor1.unsqueeze(1).repeat(1, s2, 1, 1).view(-1, *(tensor1.size()[1:]))
    mask1 = mask1.unsqueeze(1).repeat(1, s2, 1).view(-1, *(mask1.size()[1:]))
    tensor2 = tensor2.repeat(s1, 1, 1)
    mask2 = mask2.repeat(s1, 1)

    sim = sim_func(Embedding(tensor1, mask1), Embedding(tensor2, mask2))
    return sim.view(s1, s2)

def create_models(emb_dim, dropout=0.0):
    trunk = BoundingBoxTrunkModel()

    model = nn.Sequential(
        nn.Linear(trunk.output_size, emb_dim),
        Normalize(),
        nn.Dropout(p=dropout) if dropout > 0.0 else Identity()
    )
    model = BoundingBoxEmbedder(model)

    return trunk, model

def train_eval(args, train_data, dev_data, positions):
    _bbox_collate_fn = partial(bbox_collate_fn, max_bb_num=args.bb_num)

    # Create dataset & dataloader
    trans = [
        PadResize(224),
        transforms.RandomRotation(degrees=args.aug_rot),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=args.aug_erase_p,
        scale=(args.aug_erase_min, args.aug_erase_max))
    ]
    trans = transforms.Compose(trans)
    dev_trans = [
        PadResize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    dev_trans = transforms.Compose(dev_trans)

    train_dataset, train_char_idx = \
        create_datasetBB(args.root, train_data, positions,
            post_crop_transform=trans,
            collate_fn_bbox=_bbox_collate_fn,
            bbox_scale=args.bb_scale)

    train_sampler = MetricBatchSampler(
        train_dataset, train_char_idx,
        n_max_per_char = args.n_max_per_char,
        n_batch_size = args.n_batch_size,
        n_random = args.n_random
    )
    train_dataloader = DataLoader(
        train_dataset, batch_sampler=train_sampler,
        batch_size=1, num_workers=5
    )

    eval_train_dataloaders = \
        prepare_evaluation_dataloadersBB(args, args.eval_split*3, train_data, positions,
            post_crop_transform=dev_trans,
            collate_fn_bbox=_bbox_collate_fn,
            bbox_scale=args.bb_scale
        )
    eval_dev_dataloaders = \
        prepare_evaluation_dataloadersBB(args, args.eval_split, dev_data, positions,
            post_crop_transform=dev_trans,
            collate_fn_bbox=_bbox_collate_fn,
            bbox_scale=args.bb_scale
        )

    # Construct model & optimizer
    device = "cpu" if args.gpu < 0 else "cuda:{}".format(args.gpu)

    trunk, model = create_models(args.emb_dim, args.dropout)
    trunk.to(device)
    model.to(device)

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            list(trunk.parameters()) + list(model.parameters()),
            lr = args.lr, momentum = args.momentum,
            weight_decay = args.decay
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            list(trunk.parameters()) + list(model.parameters()),
            lr = args.lr,
            weight_decay = args.decay
        )
    elif args.optimizer == "RAdam":
        optimizer = RAdam(
            list(trunk.parameters()) + list(model.parameters()),
            lr = args.lr,
            weight_decay = args.decay
        )

    def lr_func(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        else:
            steps_decay = step // args.decay_freq
            return 1 / args.decay_factor ** steps_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    if args.optimizer == "RAdam":
        scheduler = None

    best_dev_eer = 1.0
    for i_epoch in range(args.epoch):
        logger.info(f"EPOCH: {i_epoch}")

        bar = tqdm(total=len(train_dataloader), smoothing=0.0)
        for (img, mask), labels in train_dataloader:
            optimizer.zero_grad()

            img, mask = img.to(device), mask.to(device)
            embedding = model(trunk([img, mask]))

            a_idx, p_idx, n_idx = get_all_triplets_indices(labels)
            if a_idx.size(0) == 0:
                logger.info("Zero triplet. Skip.")
                continue
            anchors, positives, negatives = embedding[a_idx], embedding[p_idx], embedding[n_idx]
            a_p_dist = - sim_func(anchors, positives)
            a_n_dist = - sim_func(anchors, negatives)

            dist = a_p_dist - a_n_dist
            loss_modified = dist + args.margin
            relued = torch.nn.functional.relu(loss_modified)
            num_non_zero_triplets = (relued > 0).nonzero().size(0)
            if num_non_zero_triplets > 0:
                loss =  torch.sum(relued) / num_non_zero_triplets
                loss.backward()
                optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            bar.update()
        bar.close()

        if i_epoch % args.eval_freq == 0:
            train_eer, train_eer_std = evaluate(
                args, trunk, model,
                eval_train_dataloaders, sim_func=sim_func_pair
            )
            dev_eer, dev_eer_std = evaluate(
                args, trunk, model,
                eval_dev_dataloaders, sim_func=sim_func_pair
            )
            logger.info("Train EER (mean, std):\t{}\t{}".format(train_eer, train_eer_std))
            logger.info("Eval EER (mean, std):\t{}\t{}".format(dev_eer, dev_eer_std))
            if dev_eer < best_dev_eer:
                logger.info("New best model!")
                best_dev_eer = dev_eer

                if args.save_model:
                    save_models = {
                        "trunk": trunk.state_dict(),
                        "embedder": model.state_dict(),
                        "args": [args.emb_dim, args.dropout]
                    }
                    torch.save(save_models, f"model/{args.suffix}.mdl")

    return best_dev_eer

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group("Dataset arguments")
    group.add_argument("--dataset", type=str)
    group.add_argument("--positions", type=str)
    group.add_argument("--root", type=str)
    group.add_argument("--remove-dup", action="store_true")

    group = parser.add_argument_group("Model arguments")
    group.add_argument("--emb-dim", type=int, default=500)
    group.add_argument("--dropout", type=float, default=0.0)
    group.add_argument("--bb-scale", type=float, default=1.5)
    group.add_argument("--bb-num", type=int, default=1)

    group.add_argument("--margin", type=float, default=0.1)
    group.add_argument("--smooth", action="store_true")

    group = parser.add_argument_group("Training arguments")
    group.add_argument("--epoch", type=int, default=100)
    group.add_argument("--optimizer", choices=["SGD", "Adam", "RAdam"], default="SGD")
    group.add_argument("--lr", type=float, default=0.01)
    group.add_argument("--decay", type=float, default=0.0)
    group.add_argument("--momentum", type=float, default=0.9)

    group.add_argument("--warmup", type=int, default=700)
    group.add_argument("--decay-freq", type=int, default=3000)
    group.add_argument("--decay-factor", type=int, default=2)

    group.add_argument("--n-max-per-char", type=int, default=7)
    group.add_argument("--n-batch-size", type=int, default=70)
    group.add_argument("--n-random", type=int, default=70)

    group.add_argument("--aug-erase-p", type=float, default=0.0)
    group.add_argument("--aug-erase-min", type=float, default=0.02)
    group.add_argument("--aug-erase-max", type=float, default=0.33)
    group.add_argument("--aug-rot", type=float, default=0.0)

    group.add_argument("--eval-split", type=int, default=5)
    group.add_argument("--eval-freq", type=int, default=10)

    group = parser.add_argument_group("System arguments")
    group.add_argument("--gpu", type=int, default=-1)
    group.add_argument("--suffix", type=str, default="tmp")
    group.add_argument("--save-model", action="store_true")


    args = parser.parse_args()
    args.normalize = False

    exp_label = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    formatter = "[%(asctime)s] %(levelname)s - %(name)s\t%(message)s"
    logging.basicConfig(
        level = logging.INFO,
        format = formatter,
        handlers = [logging.StreamHandler(), logging.FileHandler(f"log/{args.suffix}.log", "a")]
    )

    logger.info(str(args))

    # Load dataset
    with open(args.dataset) as h:
        data = json.load(h)
    n_split = len(data)

    with open(args.positions) as h:
        _positions = json.load(h)
    positions = {}
    for _pos in _positions:
        positions.update(_pos)
    positions = {int(key): val for key, val in positions.items()}
    # remove zero width / height bounding boxes
    positions = {
        key: [
            bbox for bbox in lst_bbox if min(bbox[2]-bbox[0], bbox[3]-bbox[1]) > 0
        ]
        for key, lst_bbox in positions.items()
    }
    positions = {key: lst_bbox for key, lst_bbox in positions.items() if len(lst_bbox) > 0}

    logger.info(f"Average number of BBox: {np.mean([len(lst_bbox) for lst_bbox in positions.values()])}")
 
    # Remove data without face positions.
    for split in data:
        for data_ind in range(len(split))[::-1]:
            _, lst_i_img = split[data_ind]
            for ind in range(len(lst_i_img))[::-1]:
                if lst_i_img[ind] not in positions:
                    del lst_i_img[ind]
            if len(lst_i_img) == 0:
                del split[data_ind]
        print(len(split))

    #
    train_data = sum(data[0:3],[])
    if args.remove_dup:
        train_data = remove_duplicate_images(train_data)
    best_eval = train_eval(args, train_data, data[3], positions)

    logger.info(f"Best evaluation result: {best_eval}")
    with open("_tmp.log", "a") as h:
        h.write("{}\t{}\t{}\n".format(best_eval, args.suffix, args))
