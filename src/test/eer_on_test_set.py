import sys
sys.path.append("src")

import argparse
from collections import defaultdict
import json

import matplotlib.pyplot as plt

import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms

from tqdm import tqdm

from model import *
from main import create_models
from util import calculate_eer, create_dataset, PairDataset

def evaluate(models, dataloader, device):
    for model in models.values():
        model.eval()

    y_score = []
    y_label = []
    with torch.no_grad():
        for data1, data2, label in tqdm(iter(dataloader), leave=False, total=len(dataloader)):
            emb1 = models["embedder"](models["trunk"](data1.to(device)))
            emb2 = models["embedder"](models["trunk"](data2.to(device)))
            #sim = torch.sum(emb1 * emb2, dim=1)
            sim = - torch.norm(emb1 - emb2, dim=1)

            y_score += sim.tolist()
            y_label += label.tolist()

    eer, results = calculate_eer(y_label, y_score)

    y_label = np.array(y_label)
    y_score = np.array(y_score)

    min_score = np.min(y_score[y_label>0.5])
    max_score = np.max(y_score[y_label<0.5])

    return eer, results, (min_score, max_score)


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--root", type=str)
    parser.add_argument("--model-fn", type=str)

    parser.add_argument("--gpu", type=int, default=-1)

    args = parser.parse_args()

    # Load dataset.
    with open(args.dataset) as h:
        data = json.load(h)
    n_split = len(data)

    test_data = data[4] # Test set of data

    trans = [
        transforms.Resize((224,224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    trans = transforms.Compose(trans)
    test_dataset, test_char_idx = create_dataset(args.root, test_data, trans)

    def create_dataloader(pairs, labels):
        dataset = PairDataset(test_dataset, pairs, labels)
        dataloader = DataLoader(dataset, batch_size=50, shuffle=False,
            collate_fn=PairDataset.collate_fn)

        return dataloader

    # Load model
    device = "cpu" if args.gpu < 0 else "cuda:{}".format(args.gpu)

    saved_models = torch.load(args.model_fn, map_location="cpu")
    trunk, model = create_models(*saved_models["args"])
    models = {"trunk": trunk, "embedder": model}
    for key in models.keys():
        models[key].load_state_dict(saved_models[key])
        models[key].to(device)

    # Compute EER
    def create_pair_uniform(char_idx, n_pair_cls=1000, seed=1234):
        u"""Uniformly sample from all positive & negative pairs"""
        np.random.seed(seed)

        img2char = {}
        for char_id, lst_img_idx in char_idx.items():
            for img_idx in lst_img_idx:
                img2char[img_idx] = char_id

        lst_img = list(img2char.keys())

        # Positive pairs
        lst_char = list(char_idx.keys())
        lst_size_char = np.array([len(char_idx[char]) for char in lst_char])
        _p = lst_size_char * (lst_size_char - 1)
        _p = _p / np.sum(_p)

        pos_pairs = []
        for char in np.random.choice(lst_char, size=n_pair_cls, p=_p):
            img1, img2 = np.random.choice(char_idx[char], size=2, replace=False).tolist()
            pos_pairs.append((img1, img2))

        # Negative pairs
        neg_pairs = []
        while len(neg_pairs) < n_pair_cls:
            img1, img2 = np.random.choice(lst_img, size=2, replace=False).tolist()
            if img2char[img1] == img2char[img2]:
                continue
            neg_pairs.append((img1, img2))

        return pos_pairs, neg_pairs, [1]*len(pos_pairs), [0]*len(neg_pairs)

    ## Uniformly samples positive and negative pairs.
    print("All")
    _ = create_pair_uniform(test_char_idx, 10000)
    all_pairs = _[0]+_[1]
    all_labels = _[2]+_[3]
    all_neg = (_[1], _[3])
    dataloader = create_dataloader(all_pairs, all_labels)

    all_eer, all_results, all_range = evaluate(models, dataloader, device)
    print(f"EER: {all_eer}")

    ## Split characters into tiers based on their frequency & sample pairs within each tier.
    dict_char_idx_tier = defaultdict(dict)
    for char_id, lst_img_idx in test_char_idx.items():
        n_img = len(lst_img_idx)

        if n_img < 5:
            tier = 1
        elif n_img < 10:
            tier = 2
        elif n_img < 50:
            tier = 3
        else:
            tier = 4

        dict_char_idx_tier[tier][char_id] = lst_img_idx

    dict_test_results = {}
    for tier, char_idx_tier in dict_char_idx_tier.items():
        print(f"Tier: {tier}")
        _ = create_pair_uniform(char_idx_tier, 10000)
        tier_pairs, tier_labels = _[0]+all_neg[0], _[2]+all_neg[1]
        dataloader = create_dataloader(tier_pairs, tier_labels)

        tier_eer, test_results, _ = evaluate(models, dataloader, device)
        print(f"EER: {tier_eer}")
        dict_test_results[tier] = test_results

    # def plot(result, color):
    #     fpr, tpr, threshold, thresh = result
    #     frr = 1. - tpr
    #     far = fpr
    #     p_frr,  = plt.plot(threshold, frr, color=color, marker="")
    #     p_far,  = plt.plot(threshold, far, color=color, marker="", ls=":")
    #     plt.axhline(all_eer, color="k", ls="--")
    #     plt.plot([thresh], [all_eer], marker="s", color="k", fillstyle="none")
    #
    #     return p_frr, p_far
    #
    # plt.ylim([-0.1, 1.1])
    # #plt.xlim(all_range)
    #
    # plt.grid()
    #
    # plt.legend(
    #     [p_frr, p_far],
    #     ["FRR", "FAR"]
    # )
    #
    # plt.xlabel("Threshold")
    # plt.ylabel("False Reject Rate (FRR) / False Acceptance Rate (FAR)")
    #
    # plt.show()
