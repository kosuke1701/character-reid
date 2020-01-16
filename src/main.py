import sys

import argparse
import datetime
import json
import logging
import time

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from pytorch_metric_learning import losses

from util import create_dataset, MetricBatchSampler, collate_fn, \
    calculate_eer
from model import EmbModel

def prepare_evaluation_dataloaders(args, n_split, data, trans):
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

def train_epoch(args, model, optimizer, dataloader):
    device = next(model.parameters()).device
    loss_func = losses.TripletMarginLoss(
        margin=args.margin, normalize_embeddings=args.normalize
    )

    model.train()

    n_batch = len(dataloader)
    lst_loss = []
    for i_batch, (batch_img, batch_label) in enumerate(dataloader):
        sys.stdout.write(f"{i_batch}/{n_batch}\r")
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
    logger = logging.getLogger("main")

    device = next(model.parameters()).device

    model.eval()

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
        logger.info(f"Dataloader EER:\t{eer}")

        _time_end_eval = time.time()
        logger.info("Computing EER took {:.2f} seconds".format(
            _time_end_eval - _time_start_eval
        ))

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

    eval_train_dataloaders = \
        prepare_evaluation_dataloaders(args, args.eval_split*3, train_data, trans)
    eval_dev_dataloaders = \
        prepare_evaluation_dataloaders(args, args.eval_split, dev_data, trans)

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
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr = args.lr,
            weight_decay = args.decay
        )
    else:
        raise NotImplementedError

    # Train & eval
    best_dev_eer = 1.0
    for i_epoch in range(args.epoch):
        logger.info(f"EPOCH {i_epoch}")
        print("Training...")
        train_loss = train_epoch(args, model, optimizer, train_dataloader)
        logger.info("Train loss:\t{}".format(train_loss))

        if i_epoch % args.eval_freq == 0:
            print("Evaluating...")
            train_eer, train_eer_std = evaluate(args, model, eval_train_dataloaders)
            dev_eer, dev_eer_std = evaluate(args, model, eval_dev_dataloaders)
            logger.info("Eval EER (mean, std):\t{}\t{}".format(train_eer, train_eer_std))
            logger.info("Eval EER (mean, std):\t{}\t{}".format(dev_eer, dev_eer_std))
            if dev_eer < best_dev_eer:
                logger.info("New best model!")
                best_dev_eer = dev_eer

    return best_dev_eer

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
    group.add_argument("--optimizer", choices=["SGD", "Adam"], default="SGD")
    group.add_argument("--lr", type=float, default=0.01)
    group.add_argument("--decay", type=float, default=0.0)
    group.add_argument("--momentum", type=float, default=0.9)

    group.add_argument("--n-max-per-char", type=int, default=10)
    group.add_argument("--n-batch-size", type=int, default=100)
    group.add_argument("--n-random", type=int, default=100)

    group.add_argument("--eval-split", type=int, default=5)
    group.add_argument("--eval-freq", type=int, default=10)

    group = parser.add_argument_group("System arguments")
    group.add_argument("--gpu", type=int, default=-1)
    group.add_argument("--suffix", type=str, default="tmp")


    args = parser.parse_args()

    exp_label = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler("log/{}.log".format(args.suffix)))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    logger.info(str(args))

    # Load dataset.
    with open(args.dataset) as h:
        data = json.load(h)
    n_split = len(data)

    #
    best_eval = train_eval(args, sum(data[0:2],[]), data[3])

    logger.info(f"Best evaluation result: {best_eval}")
    with open("_tmp.log", "a") as h:
        h.write("{}\t{}\t{}\n".format(best_eval, args.suffix, args))
