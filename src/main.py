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
from torchvision import transforms, models

from pytorch_metric_learning import losses, miners, trainers
import pytorch_metric_learning.utils.common_functions

from util import create_dataset, MetricBatchSampler, collate_fn, \
    prepare_evaluation_dataloaders, evaluate, remove_duplicate_images
from model import Identity, Normalize

def create_models(emb_dim, dropout=0.0):
    trunk = models.resnet18(pretrained=True)
    trunk_output_size = trunk.fc.in_features
    trunk.fc = Identity()

    model = nn.Sequential(
        nn.Dropout(p=dropout) if dropout > 0.0 else Identity(),
        nn.Linear(trunk_output_size, emb_dim),
        Normalize()
    )

    return trunk, model

# Override library function to use batch_sampler
def get_train_dataloader(dataset, batch_size, sampler, num_workers, collate_fn):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False
    )
pytorch_metric_learning.utils.common_functions.get_train_dataloader = get_train_dataloader

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
    # number of batches given to trainer
    n_batch = int(len(train_dataloader))

    eval_train_dataloaders = \
        prepare_evaluation_dataloaders(args, args.eval_split*3, train_data, trans)
    eval_dev_dataloaders = \
        prepare_evaluation_dataloaders(args, args.eval_split, dev_data, trans)

    # Construct model & optimizer
    device = "cpu" if args.gpu < 0 else "cuda:{}".format(args.gpu)

    trunk, model = create_models(args.emb_dim, args.dropout)
    trunk.to(device)
    model.to(device)

    if args.metric_loss == "triplet":
        loss_func = losses.TripletMarginLoss(
            margin=args.margin, normalize_embeddings=args.normalize,
            smooth_loss=args.smooth
        )
    elif args.metric_loss == "arcface":
        loss_func = losses.ArcFaceLoss(
            margin = args.margin,
            num_classes=len(train_data),
            embedding_size=args.emb_dim
        )
        loss_func.to(device)

    if args.optimizer == "SGD":
        trunk_optimizer = torch.optim.SGD(
            trunk.parameters(),
            lr = args.lr, momentum = args.momentum,
            weight_decay = args.decay
        )
        model_optimizer = torch.optim.SGD(
            model.parameters(),
            lr = args.lr, momentum = args.momentum,
            weight_decay = args.decay
        )
        optimizers = {
            "trunk_optimizer": trunk_optimizer,
            "embedder_optimizer": model_optimizer
        }
        if args.metric_loss == "arcface":
            loss_optimizer = torch.optim.SGD(
                loss_func.parameters(), lr=args.lr, momentum=args.momentum,
                weight_decay = args.decay
            )
            optimizers["loss_optimizer"] = loss_optimizer
    elif args.optimizer == "Adam":
        trunk_optimizer = torch.optim.Adam(
            trunk.parameters(),
            lr = args.lr,
            weight_decay = args.decay
        )
        model_optimizer = torch.optim.Adam(
            model.parameters(),
            lr = args.lr,
            weight_decay = args.decay
        )
        optimizers = {
            "trunk_optimizer": trunk_optimizer,
            "embedder_optimizer": model_optimizer
        }
        if args.metric_loss == "arcface":
            loss_optimizer = torch.optim.Adam(
                loss_func.parameters(), lr=args.lr,
                weight_decay = args.decay
            )
            optimizers["loss_optimizer"] = loss_optimizer
    else:
        raise NotImplementedError

    def lr_func(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        else:
            steps_decay = step // args.decay_freq
            return 1 / args.decay_factor ** steps_decay

    trunk_scheduler = torch.optim.lr_scheduler.LambdaLR(trunk_optimizer, lr_func)
    model_scheduler = torch.optim.lr_scheduler.LambdaLR(model_optimizer, lr_func)
    schedulers = {"trunk_scheduler": trunk_scheduler, "model_scheduler": model_scheduler}

    if args.miner == "none":
        mining_funcs = {}
    elif args.miner == "batch-hard":
        mining_funcs = {
            "post_gradient_miner": miners.BatchHardMiner(use_similarity=True)
        }

    best_dev_eer = 1.0
    i_epoch = 0
    def end_of_epoch_hook(trainer):
        nonlocal i_epoch, best_dev_eer

        logger.info(f"EPOCH\t{i_epoch}")

        if i_epoch % args.eval_freq == 0:
            train_eer, train_eer_std = evaluate(
                args, trainer.models["trunk"], trainer.models["embedder"],
                eval_train_dataloaders
            )
            dev_eer, dev_eer_std = evaluate(
                args, trainer.models["trunk"], trainer.models["embedder"],
                eval_dev_dataloaders
            )
            logger.info("Eval EER (mean, std):\t{}\t{}".format(train_eer, train_eer_std))
            logger.info("Eval EER (mean, std):\t{}\t{}".format(dev_eer, dev_eer_std))
            if dev_eer < best_dev_eer:
                logger.info("New best model!")
                best_dev_eer = dev_eer

        i_epoch += 1

    def end_of_iteration_hook(trainer):
        for scheduler in schedulers.values():
            scheduler.step()

    trainer = trainers.MetricLossOnly(
        models = {"trunk": trunk, "embedder": model},
        optimizers = optimizers,
        batch_size = None,
        loss_funcs = {"metric_loss": loss_func},
        mining_funcs = mining_funcs,
        iterations_per_epoch = n_batch,
        dataset = train_dataset,
        data_device = None,
        loss_weights = None,
        sampler = train_sampler,
        collate_fn = collate_fn,
        lr_schedulers = None,
        end_of_epoch_hook = end_of_epoch_hook,
        end_of_iteration_hook=end_of_iteration_hook,
        dataloader_num_workers=1
    )

    trainer.train(num_epochs=args.epoch)

    if args.save_model:
        save_models = {
            "trunk": trainer.models["trunk"].state_dict(),
            "embedder": trainer.models["embedder"].state_dict(),
            "args": [args.emb_dim]
        }
        torch.save(save_models, f"model/{args.suffix}.mdl")

    return best_dev_eer

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group("Dataset arguments")
    group.add_argument("--dataset", type=str)
    group.add_argument("--root", type=str)
    group.add_argument("--remove-dup", action="store_true")

    group = parser.add_argument_group("Model arguments")
    group.add_argument("--emb-dim", type=int, default=500)
    group.add_argument("--normalize", action="store_true")
    group.add_argument("--dropout", type=float, default=0.0)

    group.add_argument("--metric-loss", choices=["triplet", "arcface"], default="triplet")
    group.add_argument("--margin", type=float, default=0.1)
    group.add_argument("--smooth", action="store_true")

    group.add_argument("--miner", type=str,
        choices=["none", "batch-hard"], default="none")
    group.add_argument("--type-of-triplets", type=str,
        choices=["all", "hard", "semihard"], default="all")

    group = parser.add_argument_group("Training arguments")
    group.add_argument("--epoch", type=int, default=100)
    group.add_argument("--optimizer", choices=["SGD", "Adam"], default="SGD")
    group.add_argument("--lr", type=float, default=0.01)
    group.add_argument("--decay", type=float, default=0.0)
    group.add_argument("--momentum", type=float, default=0.9)

    group.add_argument("--warmup", type=int, default=700)
    group.add_argument("--decay-freq", type=int, default=3000)
    group.add_argument("--decay-factor", type=int, default=2)

    group.add_argument("--n-max-per-char", type=int, default=7)
    group.add_argument("--n-batch-size", type=int, default=70)
    group.add_argument("--n-random", type=int, default=70)

    group.add_argument("--eval-split", type=int, default=5)
    group.add_argument("--eval-freq", type=int, default=10)

    group = parser.add_argument_group("System arguments")
    group.add_argument("--gpu", type=int, default=-1)
    group.add_argument("--suffix", type=str, default="tmp")
    group.add_argument("--save-model", action="store_true")


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
    train_data = sum(data[0:3],[])
    if args.remove_dup:
        train_data = remove_duplicate_images(train_data)
    best_eval = train_eval(args, train_data, data[3])

    logger.info(f"Best evaluation result: {best_eval}")
    with open("_tmp.log", "a") as h:
        h.write("{}\t{}\t{}\n".format(best_eval, args.suffix, args))
