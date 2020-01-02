import sys

import argparse
import datetime
import json
import logging

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from pytorch_metric_learning import losses

from util import create_dataset, MetricBatchSampler, collate_fn, \
    calculate_eer
from model import EmbModel

def train_epoch(args, model, optimizer, dataloader):
    device = next(model.parameters()).device
    loss_func = losses.TripletMarginLoss(
        margin=args.margin, normalize_embeddings=args.normalize
    )
    
    model.train()

    lst_loss = []
    for i_batch, (batch_img, batch_label) in enumerate(dataloader):
        sys.stdout.write(f"{i_batch}\r")
        sys.stdout.flush()

        optimizer.zero_grad()

        batch_img = batch_img.to(device)
        batch_label = batch_label.to(device)

        embeddings = model(batch_img)
        loss = loss_func(embeddings, batch_label)
        lst_loss.append(loss.item())

        loss.backward()
        optimizer.step()
    
    return np.mean(lst_loss)

def evaluate(args, model, dataloaders):
    device = next(model.parameters()).device

    model.eval()

    lst_eer = []
    for i_loader, dataloader in enumerate(dataloaders):
        print(f"Dataloader {i_loader}")
        with torch.no_grad():
            embeddings = []
            labels = []

            print("Computing embeddings...")
            n_batch = len(dataloader)
            for i_batch, (batch_img, batch_label) in enumerate(dataloader):
                sys.stdout.write(f"{i_batch}/{n_batch}\r")
                sys.stdout.flush()

                batch_img = batch_img.to(device)
                embs = model(batch_img)

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

    return np.mean(lst_eer), np.std(lst_eer)

def train_eval(args, train_data, dev_data):
    logger = logging.getLogger("main")
    # Create dataset & dataloader
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

    train_dataset, train_char_idx = \
        create_dataset(args.root, train_data, trans)

    train_sampler = MetricBatchSampler(
        train_dataset, train_char_idx,
        n_max_per_char = args.n_max_per_char,
        n_batch_size = args.n_batch_size,
        n_random = args.n_random
    )
    train_dataloader = DataLoader(
        train_dataset, batch_sampler=train_sampler,
        collate_fn = collate_fn
    )

    dev_dataloaders = []
    for i in range(args.eval_split):
        dev_dataset, dev_char_idx = \
            create_dataset(args.root, dev_data[i::args.eval_split], trans)
        dev_dataloader = DataLoader(
            dev_dataset, batch_size=100,
            collate_fn = collate_fn, shuffle=True
        )
        dev_dataloaders.append(dev_dataloader)

    # Construct model & optimizer
    device = "cpu" if args.gpu < 0 else "cuda:{}".format(args.gpu)

    model = EmbModel(args.emb_dim, 1)
    model.to(device)

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr = args.lr, momentum = args.momentum,
            weight_decay = args.decay
        )
    else:
        raise NotImplementedError
    
    # Train & eval
    for i_epoch in range(args.epoch):
        logger.info(f"EPOCH {i_epoch}")
        print("Training...")
        train_loss = train_epoch(args, model, optimizer, train_dataloader)
        logger.info("Train loss:\t{}".format(train_loss))

        print("Evaluating...")
        dev_eer = evaluate(args, model, dev_dataloaders)
        logger.info("Eval EER (mean, std):\t{}\t{}".format(dev_eer[0], dev_eer[1]))



if __name__=="__main__":
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group("Dataset arguments")
    group.add_argument("--dataset", type=str)
    group.add_argument("--root", type=str)

    group = parser.add_argument_group("Model arguments")
    group.add_argument("--emb-dim", type=int, default=500)
    group.add_argument("--normalize", action="store_true")
    group.add_argument("--margin", type=float, default=0.1)

    group = parser.add_argument_group("Training arguments")
    group.add_argument("--epoch", type=int, default=100)
    group.add_argument("--optimizer", choices=["SGD"], default="SGD")
    group.add_argument("--lr", type=float, default=0.01)
    group.add_argument("--decay", type=float, default=0.0)
    group.add_argument("--momentum", type=float, default=0.9)
    
    group.add_argument("--n-max-per-char", type=int, default=10)
    group.add_argument("--n-batch-size", type=int, default=100)
    group.add_argument("--n-random", type=int, default=100)

    group.add_argument("--eval-split", type=int, default=5)

    group = parser.add_argument_group("System arguments")
    group.add_argument("--gpu", type=int, default=-1)


    args = parser.parse_args()

    exp_label = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler("log/{}.log".format(exp_label)))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    logger.info(str(args))

    # Load dataset.
    with open(args.dataset) as h:
        data = json.load(h)
    n_split = len(data)

    #
    train_eval(args, data[0], data[1])

