import sys
sys.path.append("src")

import argparse
from collections import defaultdict
import json

import matplotlib.pyplot as plt

import numpy as np

from sklearn.cluster import AgglomerativeClustering

import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms

from tqdm import tqdm

from model import *
from main import create_models
from util import calculate_eer, create_dataset, remove_duplicate_images, collate_fn

def macro_precision(cls2img, img2ent):
    n_correct = 0
    for lst_img in cls2img:
        if len(set([img2ent[i_img] for i_img in lst_img])) == 1:
            n_correct += 1
    return n_correct / len(cls2img)

def micro_precision(cls2img, img2ent):
    n_correct = 0
    n_all_img = 0
    for lst_img in cls2img:
        per_ent_count = defaultdict(int)
        for i_img in lst_img:
            per_ent_count[img2ent[i_img]] += 1

        n_correct += max(list(per_ent_count.values()))
        n_all_img += len(lst_img)

    return n_correct / n_all_img

def pairwise_precision_recall(cls2img, img2ent, ent2img):
    n_pairs = 0
    n_hits = 0

    for lst_img in cls2img:
        size_cls = len(lst_img)
        n_pairs += size_cls * (size_cls-1) // 2

        per_ent_count = defaultdict(int)
        for i_img in lst_img:
            per_ent_count[img2ent[i_img]] += 1
        for size_ent in per_ent_count.values():
            if size_ent > 1:
                n_hits += size_ent * (size_ent - 1) // 2

    n_ent_pairs = 0
    for lst_img in ent2img:
        size_ent = len(lst_img)

        n_ent_pairs += size_ent * (size_ent - 1) // 2

    # precision, recall
    if n_pairs == 0:
        prc = 1.0
    else:
        prc = n_hits / n_pairs
    rcl = n_hits / n_ent_pairs
    return prc, rcl

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
    #test_data = test_data[-100:]#For DEBUG
    test_data = remove_duplicate_images(test_data)

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
    n_img = len(test_dataset)

    labels = test_dataset.lst_label
    old2new = {i_old: i_new for i_new, i_old in enumerate(set(labels))}
    labels = [old2new[_] for _ in labels] # img2char
    char2samples = [[] for _ in range(len(set(labels)))] # char2img
    for i_img, label in enumerate(labels):
        char2samples[label].append(i_img)



    print(f"Number of test images: {n_img}")

    # Load model
    device = "cpu" if args.gpu < 0 else "cuda:{}".format(args.gpu)

    saved_models = torch.load(args.model_fn, map_location="cpu")
    trunk, model = create_models(*saved_models["args"])
    models = {"trunk": trunk, "embedder": model}
    for key in models.keys():
        models[key].load_state_dict(saved_models[key])
        models[key].to(device)

    # Compute distance matrix
    s_batch = 600
    dist_matrix = np.zeros((n_img, n_img))

    print("Computing distance matrix...")
    n_batch = int(np.ceil(n_img / s_batch))
    progress = tqdm(total=n_batch*(n_batch-1)//2+n_batch)
    for i_img in range(0, n_img, s_batch):
        for j_img in range(0, i_img+s_batch, s_batch):# We assume symmetric distance.
            lst_img1 = [test_dataset[_] for _ in range(i_img, min(i_img+s_batch, n_img))]
            lst_img2 = [test_dataset[_] for _ in range(j_img, min(j_img+s_batch, n_img))]

            lst_img1, _ = collate_fn(lst_img1)
            lst_img2, _ = collate_fn(lst_img2)

            lst_img1 = lst_img1.to(device)
            lst_img2 = lst_img2.to(device)

            with torch.no_grad():
                emb1, emb2 = map(lambda x: models["embedder"](models["trunk"](x)),
                    [lst_img1, lst_img2])
                dist = torch.norm(emb1.unsqueeze(1) - emb2.unsqueeze(0), dim=2)
                dist = np.array(dist.tolist())

            dist_matrix[i_img:i_img+s_batch, j_img:j_img+s_batch] = dist
            dist_matrix[j_img:j_img+s_batch, i_img:i_img+s_batch] = dist.T

            progress.update()
    progress.close()

    # Hierarchical Clustering
    cls_model = AgglomerativeClustering(distance_threshold=0, n_clusters=None,
        affinity="precomputed", linkage="average")
    cls_model.fit(dist_matrix)

    # Evaluate clustering results.
    def evaluate_clusters(clusters):
        cls2data = clusters.values()
        data2cls = [None]*n_img
        for i_cls, cls in enumerate(cls2data):
            for i_img in cls:
                data2cls[i_img] = i_cls
        truecls2data = char2samples
        data2truecls = labels

        macro_prc = macro_precision(cls2data, data2truecls)
        macro_rcl = macro_precision(truecls2data, data2cls)
        micro_prc = micro_precision(cls2data, data2truecls)
        micro_rcl = micro_precision(truecls2data, data2cls)
        pairwise_prc, pairwise_rcl = pairwise_precision_recall(cls2data, data2truecls, truecls2data)

        return macro_prc, macro_rcl, micro_prc, micro_rcl, pairwise_prc, pairwise_rcl

    depth = cls_model.children_.shape[0]
    print(f"Total number of merges: {depth}")
    current_clusters = {i_img: [i_img] for i_img in range(n_img)}
    result = [evaluate_clusters(current_clusters)]
    steps = [0]
    progress = tqdm(total=depth)
    for i_step, (cls1, cls2) in enumerate(cls_model.children_):
        current_clusters[n_img + i_step] = current_clusters[cls1] + current_clusters[cls2]
        del current_clusters[cls1], current_clusters[cls2]
        if (i_step+1) % max(depth//100,1) == 0:
            result.append(evaluate_clusters(current_clusters))
            steps.append(i_step+1)
        progress.update()
    progress.close()
    result = np.array(result)

    # Macro precision & recall
    plt.plot(steps, result[:,0], c="r", ls="-", label="Macro precision")
    plt.plot(steps, result[:,1], c="r", ls=":", label="Macro recall")
    # Micro precision & recall
    plt.plot(steps, result[:,2], c="b", ls="-", label="Micro precision")
    plt.plot(steps, result[:,3], c="b", ls=":", label="Micro recall")
    # Pairwise precision & recall
    plt.plot(steps, result[:,4], c="g", ls="-", label="Pairwise precision")
    plt.plot(steps, result[:,5], c="g", ls=":", label="Pairwise recall")

    plt.xlabel("Clustering steps")
    plt.ylim([-0.1,1.0])

    plt.grid()
    plt.legend()

    plt.show()


    print("Done.")
