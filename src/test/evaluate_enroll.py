import sys
sys.path.append("src")

import argparse
from collections import defaultdict
import json

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from matplotlib import cm

import numpy as np

from scipy.interpolate import interp1d

from sklearn.metrics import precision_recall_curve
from sklearn.utils import shuffle

import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms

from tqdm import tqdm

from model import *
from util import calculate_eer, create_dataset, remove_duplicate_images

def split_enroll_target(char_idx, n_enroll=10, min_nimg=20):
    enrolls = []
    targets = []
    for i_char, lst_i_img in char_idx.items():
        if len(lst_i_img) < min_nimg:
            continue

        lst_i_img = shuffle(lst_i_img)

        enrolls.append(lst_i_img[:n_enroll])
        targets.append(lst_i_img[n_enroll:])

    return enrolls, targets

def get_image_tensors(dataset, lst_i_img):
    data = [dataset[idx][0].unsqueeze(0) for idx in lst_i_img]
    data = torch.cat(data)

    return data

def evaluate(models, dataset, enrolls, targets, s_batch=200, mode="Any"):
    device = next(models["embedder"].parameters()).device

    for model in models.values():
        model.eval()

    with torch.no_grad():
        emb_targets = []
        for lst_i_img in targets:
            embs = []
            for i_start in range(0, len(lst_i_img), s_batch):
                tensor = get_image_tensors(dataset, lst_i_img[i_start:i_start+s_batch]).to(device)
                emb = models["embedder"](models["trunk"](tensor))
                embs.append(emb)
            embs = torch.cat(embs, dim=0)
            emb_targets.append(embs)

        if mode == "Avg":
            # Compute embeddings of enroll and target images.
            emb_enrolls = []
            for lst_i_img in enrolls:
                tensor = get_image_tensors(dataset, lst_i_img).to(device)
                emb = models["embedder"](models["trunk"](tensor))
                emb_enrolls.append(emb)
            mean_emb_enroll = [
                torch.mean(emb, dim=0, keepdim=True) for emb in emb_enrolls
            ]
            mean_emb_enroll = torch.cat(mean_emb_enroll, dim=0)

            # For each target voice, compute similiarity scores against enroll embeddings.
            # We only consider maximum scores for each target voice.
            scores_target = []
            labels_target = []
            for true_ind, emb in enumerate(emb_targets):
                #scores = torch.sum(mean_emb_enroll.unsqueeze(0) * emb.unsqueeze(1), dim=2)
                scores = - torch.norm(mean_emb_enroll.unsqueeze(0) - emb.unsqueeze(1), dim=2)
                # i_target x i_enroll
                max_scores, max_ind = torch.max(scores, dim=1)

                scores = max_scores.tolist()
                labels = [1 if pred_ind == true_ind else 0 for pred_ind in max_ind.tolist()]

                scores_target.append(scores)
                labels_target.append(labels)
        elif mode == "Any":
            # Compute embeddings of enroll and target images.
            emb_enrolls = []
            for lst_i_img in enrolls:
                tensor = get_image_tensors(dataset, lst_i_img).to(device)
                emb = models["embedder"](models["trunk"](tensor))
                emb_enrolls.append(emb.unsqueeze(0))
            mean_emb_enroll = torch.cat(emb_enrolls, dim=0)
            # i_char x i_enroll x dim

            # For each target voice, compute similiarity scores against enroll embeddings.
            # We only consider maximum scores for each target voice.
            scores_target = []
            labels_target = []
            for true_ind, emb in enumerate(emb_targets):
                # emb: i_target x dim
                lst_scores = []
                for i in range(mean_emb_enroll.size(1)):
                    #scores = torch.sum(mean_emb_enroll[:,i,:] * emb.unsqueeze(1), dim=2)
                    scores = - torch.norm(mean_emb_enroll[:,i,:] - emb.unsqueeze(1), dim=2)
                    lst_scores.append(scores.unsqueeze(2))
                scores = torch.cat(lst_scores, dim=2)
                # i_target x i_char x i_enroll
                scores, _ = torch.max(scores, dim=2)
                max_scores, max_ind = torch.max(scores, dim=1)

                scores = max_scores.tolist()
                labels = [1 if pred_ind == true_ind else 0 for pred_ind in max_ind.tolist()]

                scores_target.append(scores)
                labels_target.append(labels)

    # Compute Top1 metrics
    global_top1 = np.mean(sum(labels_target, []))
    mean_top1 = np.mean([np.mean(labels) for labels in labels_target])

    print(f"Global Top1: {global_top1:.4f}")
    print(f"Mean Top1: {mean_top1:.4f}")

    # Compute precision recalls.
    prc, rcl, thr = precision_recall_curve(sum(labels_target, []), sum(scores_target, []))
    prc = [np.mean(labels)] + prc.tolist() + [1.0]
    rcl = [1] + rcl.tolist() + [0.0]
    thr = [-1.0] + thr.tolist() + [1.0]
    global_pr = (prc, rcl, thr)

    local_prs = []
    for scores, labels in zip(scores_target, labels_target):
        prc, rcl, thr = precision_recall_curve(
            labels, scores
        )
        prc = [np.mean(labels)] + prc.tolist() + [1.0]
        rcl = [1] + rcl.tolist() + [0.0]
        thr = [-1.0] + thr.tolist() + [1.0]
        local_prs.append((prc, rcl, thr))

    min_thr = min(*list(map(np.min, [global_pr[2]] + list(map(lambda x:x[2], local_prs)))))
    max_thr = max(*list(map(np.max, [global_pr[2]] + list(map(lambda x:x[2], local_prs)))))

    global_prc = interp1d(global_pr[2], global_pr[0][:-1], kind="previous", fill_value=global_top1)
    global_rcl = interp1d(global_pr[2], global_pr[1][:-1], kind="next", fill_value=0.0)

    lst_local_prc = []
    lst_local_rcl = []
    for local_pr, labels in zip(local_prs, labels_target):
        if len(labels) < 2:
            print(labels)
        if sum(labels) < 1:
            prc = lambda thr: 0.0*thr
            rcl = lambda thr: 0.0*thr
        else:
            prc = interp1d(local_pr[2], local_pr[0][:-1], kind="previous",
                fill_value=np.mean(labels), bounds_error=False)
            rcl = interp1d(local_pr[2], local_pr[1][:-1], kind="next",
                fill_value=0.0, bounds_error=False)
        lst_local_prc.append(prc)
        lst_local_rcl.append(rcl)

    thr_new = np.linspace(min_thr, max_thr, 200)

    prc = [_(thr_new).tolist() for _ in lst_local_prc]
    prc = np.array(prc)
    prc = np.mean(prc, axis=0)

    rcl = [_(thr_new).tolist() for _ in lst_local_rcl]
    rcl = np.array(rcl)
    rcl = np.mean(rcl, axis=0)

    return (global_top1, mean_top1), (global_prc(thr_new), global_rcl(thr_new), prc, rcl)



if __name__=="__main__":
    # TODO: Avgの時のlocalのprecision recallがおかしい気がするのでバグじゃないか調査する
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--root", type=str)
    parser.add_argument("--model-fn", type=str)

    parser.add_argument("--n-enroll", type=int, nargs="*")
    parser.add_argument("--min-freq", type=int)

    parser.add_argument("--mode", choices=["Any", "Avg"])

    parser.add_argument("--gpu", type=int, default=-1)

    args = parser.parse_args()

    # Load dataset.
    with open(args.dataset) as h:
        data = json.load(h)
    n_split = len(data)

    test_data = data[4] # Test set of data
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

    # Load model
    device = "cpu" if args.gpu < 0 else "cuda:{}".format(args.gpu)

    models = torch.load(args.model_fn, map_location="cpu")
    for model in models.values():
        model.to(device)

    colormap = plt.get_cmap("hsv")
    plots = []
    for enroll, color in zip(args.n_enroll, np.linspace(0,0.8,len(args.n_enroll))):
        print("Enroll", enroll)
        enrolls, targets = split_enroll_target(test_char_idx, n_enroll=enroll, min_nimg=args.min_freq)
        print(f"Found {len(enrolls)} characters.")
        tops, prrcs = evaluate(models, test_dataset, enrolls, targets, mode=args.mode)

        glpr, glrc, lopr, lorc = prrcs
        p_glo, = plt.plot(glrc, glpr, color=colormap(color), ls="-")
        p_loc, = plt.plot(lorc, lopr, color=colormap(color), ls=":")
        plots.append((p_glo, p_loc))
    plt.grid()
    plt.legend(plots, list(map(str, args.n_enroll)),
    handler_map={tuple: HandlerTuple(ndivide=None)})

    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.show()
