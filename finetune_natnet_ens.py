import argparse
import os
import json
from pathlib import Path
import numpy as np
import random
import yaml

import torch

from NAT_extended.data_providers.providers_factory import get_data_provider
from NAT_extended.search.search_spaces.search_spaces_factory import get_search_space
from NAT_extended.supernets.supernets_factory import get_supernet
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics
from NAT_extended.evaluation.evaluate import validate_ee
from OFA_mbv3_extended.train.early_stopping import LossEarlyStoppingMeter
from ofa.utils import get_net_device
from tqdm import tqdm
from NAT_extended.evaluation.utils import *

from NAT_extended.train.optimizers import Ranger


def parser_add_arguments(parser):
    parser.add_argument(
        "--subnet_path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--exp",
        type=int,
        required=True
    )

    parser.add_argument(
        "--ordering",
        type=str,
        choices=["DESC", "ASC", "UNIF", "MIX"],
        required=True
    )

    parser.add_argument(
        "--lre",
        type=int,
        choices=[4, 5],
        required=True
    )

    parser.add_argument(
        "--opt",
        type=str,
        choices=["sgd", "adamw", "ranger"],
        required=True
    )

    parser.add_argument(
        "--nw",
        type=int,
        default=8,
        required=True
    )

    parser.add_argument(
        "--search_type",
        type=str,
        default="ea"
    )

    args = parser.parse_args()

    assert os.path.isfile(args.subnet_path)

    args.dataset_path = None

    return args


def assign_dataset_arguments(args):
    if args.dataset == "tiny_imagenet":
        args.valid_size = 0.15
        args.resize_scale = 0.85
    elif args.dataset == "imagenet":
        args.valid_size = 100000
        args.resize_scale = 0.08
    elif args.dataset == "cifar10":
        args.valid_size = 5000
        args.resize_scale = 1.0
    elif args.dataset == "cifar100":
        args.valid_size = 5000
        args.resize_scale = 1.0
    elif args.dataset == "cinic10":
        args.valid_size = None  # use default val dataset
        args.resize_scale = 1.0
    elif args.dataset == "aircraft":
        args.valid_size = None  # use default val dataset
        args.resize_scale = 0.35
    elif args.dataset == "cars":
        args.valid_size = 0.15
        args.resize_scale = 0.25
    elif args.dataset == "dtd":
        args.valid_size = None  # use default val dataset
        args.resize_scale = 0.2
    elif args.dataset == "flowers102":
        args.valid_size = 1000
        args.resize_scale = 1.0
    elif args.dataset == "food101":
        args.valid_size = 0.15
        args.resize_scale = 1.0
    elif args.dataset == "pets":
        args.valid_size = 0.15
        args.resize_scale = 1.0
    elif args.dataset == "stl10":
        args.valid_size = 0.15
        args.resize_scale = 0.75
    else:
        raise NotImplementedError


def assign_training_arguments(args):
    args.manual_seed = 0

    args.workers = args.nw
    args.patience = 30

    args.train_batch_size = 64  # input batch size for training
    args.test_batch_size = 512  # input batch size for testing

    args.epochs = 150

    if args.lre == 4:
        args.init_lr = 1e-4
    if args.lre == 5:
        args.init_lr = 1e-5

    args.wd = 5e-4  # optimizer weight decay
    args.momentum = 0.9

    args.T_max = args.epochs
    args.eta_min = 1e-8

    args.dropout_rate = 0


def get_weights(n_exits, ordering):
    if ordering == "UNIF":
        n = 1 / n_exits
        branches_weights = [round(n, 4) for _ in range(0, n_exits)]
        ensemble_weights = branches_weights.copy()

    else:
        m = sum(range(1, n_exits + 1))
        n = 1 / m

        branches_weights = [round(n * i, 4) for i in range(1, n_exits + 1)]
        ensemble_weights = branches_weights.copy()

        if ordering == "MIX" or ordering == "DESC":
            branches_weights = sorted(branches_weights, reverse=True)
            if ordering == "DESC":
                ensemble_weights = sorted(ensemble_weights, reverse=True)

    return branches_weights, ensemble_weights


def train(epochs, net, data_provider, criterion, optimizer, scheduler, log_file, save_path, patience, step, branches_weights, ensemble_weights):

    best_acc = 0
    best_epoch = 0
    early_stopping_meter = LossEarlyStoppingMeter(patience=patience)
    model_best_path = os.path.join(save_path, f"model_best_step{step}.pth.tar")

    for epoch in range(1, epochs + 1):

        train_loss, (train_top1, train_top5) = train_one_epoch(
            epoch,
            net,
            data_provider.train,
            criterion,
            optimizer,
            scheduler,
            branches_weights,
            ensemble_weights
        )
        valid_provider = data_provider.valid if data_provider.valid is not None else data_provider.test
        valid_loss, (valid_top1, valid_top5), _, _ = validate_ee(
            net,
            valid_provider,
            criterion,
            branches_weights,
            ensemble_weights,
            epoch-1
        )

        if valid_top1 > best_acc:
            best_acc = valid_top1
            best_epoch = epoch
            save_dict = {
                "epoch": epoch,
                "acc": best_acc,
                "optimizer": optimizer.state_dict(),
                "state_dict": net.state_dict(),
            }
            torch.save(save_dict, model_best_path)

        train_str = f"TRAIN epoch [{epoch}/{epochs}]\t" \
                    f"loss={train_loss:.3f}, top1={train_top1:.3f}, top5={train_top5:.3f}"
        valid_str = f"VALIDATE epoch [{epoch}/{epochs}]\t" \
                    f"loss={valid_loss:.3f}, top1={valid_top1:.3f} ({best_acc:.3f} @{best_epoch}), top5={valid_top5:.3f}\t"

        log_str = train_str + "\n" + valid_str + "\n"
        log_str += "--------------------------------------------------------------------------------------------" + "\n"
        log_file.write(log_str)

        should_stop = early_stopping_meter.update(valid_loss)
        if should_stop:
            return best_epoch

    return best_epoch


def train_one_epoch(epoch, net, train_loader, criterion, optimizer, scheduler, bw, ew):
    # switch to train mode
    net.train()
    device = get_net_device(net)

    net_loss_meter = AverageMeter()
    net_metric_dict = get_metric_dict()

    n_batches = len(train_loader)

    with tqdm(total=len(train_loader), desc="Train Epoch #{}".format(epoch)) as t:
        for batch, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # compute output and loss
            outputs = net(images)

            # compute loss for each branch
            branches_losses = [criterion(output, labels) for output in outputs]

            weighted_losses = []
            for branch_loss, branch_weight in zip(branches_losses, bw):
                weighted_losses.append(branch_loss * branch_weight)

            # add the losses together
            net_loss = 0
            for weighted_loss in weighted_losses:
                net_loss += weighted_loss

            weighted_outputs = []
            for output, weight in zip(outputs, ew):
                weighted_outputs.append(output * weight)

            # use the weighted outputs and sum to obtain the net output
            net_output = torch.stack(weighted_outputs)
            net_output = torch.sum(net_output, dim=0)

            # measure accuracy and record loss
            net_loss_meter.update(net_loss.item(), images.size(0))
            update_metric(net_metric_dict, net_output, labels)

            optimizer.zero_grad()
            net_loss.backward()
            optimizer.step()
            scheduler.step((epoch-1) + batch / n_batches)

            t.set_postfix({
                "loss": net_loss_meter.avg,
                **get_metric_vals(net_metric_dict, return_dict=True),
                "lr": scheduler.get_last_lr()
            })
            t.update(1)

    return net_loss_meter.avg, get_metric_vals(net_metric_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='finetune NATnets')
    args = parser_add_arguments(parser)

    work_dir, ft_subnet = os.path.split(args.subnet_path)
    ft_subnet = ft_subnet[:-5]

    nat_exp_dir = Path(work_dir).parent
    args.network = os.path.split(Path(work_dir).parent.parent.parent)[1]
    args.dataset = os.path.split(Path(work_dir).parent.parent.parent.parent)[1][3:]

    assign_dataset_arguments(args)
    assign_training_arguments(args)

    # Cache the args as a text string to save them in the output dir
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    # set seeds
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    save_path = os.path.join(work_dir, "finetune", ft_subnet, "exp_"+str(args.exp))
    os.makedirs(save_path, exist_ok=True)

    # convert net string into valid dictionary and extract the data
    with open(args.subnet_path, "r") as subnet_file:
        subnet_dict = json.load(subnet_file)
        subnet_arch = subnet_dict["arch"]
        resolution = subnet_arch.pop("r")

    for step in [1, 2]:

        # get the dataloaders for the dataset
        data_provider = get_data_provider(
            dataset=args.dataset,
            save_path=args.dataset_path,
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            valid_size=args.valid_size,
            n_worker=args.workers,
            image_size=resolution,  # only care about the size specified in the NAT obtained architectures
            resize_scale=args.resize_scale,
            distort_color=None,
            num_replicas=None,
            rank=None
        )

        # build the search space for the supernet
        search_space = get_search_space(
            net_name=args.network,
            search_type=args.search_type,
            image_scale_list=resolution
        )
        # construct the supernet given the search space
        supernet = get_supernet(
            net_name=args.network,
            n_classes=data_provider.n_classes,
            dropout_rate=args.dropout_rate,
            search_space=search_space
        )

        # load the weights from the checkpoint
        supernet_weights = os.path.join(nat_exp_dir, "supernet", "checkpoint_epoch@150.pth.tar")
        supernet_state_dicts = torch.load(supernet_weights, map_location='cpu')
        state_dicts = [
            supernet_state_dicts['model_w1.0_state_dict'],
            supernet_state_dicts['model_w1.2_state_dict']
        ]
        supernet.load_state_dict(state_dicts)

        # create and open log file
        log_file_path = os.path.join(save_path, "log.txt")
        file_mode = "w" if step == 1 else "a"
        log_file = open(os.path.join(log_file_path), file_mode)

        log_file.write(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> STEP {step} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")

        # extract the subnet with the structure previously retrieved, and set the running statistics
        supernet.set_active_subnet(**subnet_arch)

        subnet = supernet.get_active_some_exits_subnet(list(range(1, subnet_arch["nd"]+1)), preserve_weight=True)
        branches_weights, ensemble_weights = get_weights(subnet_arch["nd"], args.ordering)

        sdl = data_provider.build_sub_train_loader(data_provider.subset_size, data_provider.subset_batch_size)
        set_running_statistics(subnet, sdl)

        print("branches weights:", branches_weights)
        print("ensemble weights:", ensemble_weights)
        print(subnet.module_str)

        # move network to gpu and define the loss calculation criterion
        subnet = subnet.cuda()
        criterion = torch.nn.CrossEntropyLoss()

        if args.opt == "sgd":
            optimizer = torch.optim.SGD(subnet.parameters(), lr=args.init_lr, weight_decay=args.wd, momentum=args.momentum)
        if args.opt == "adamw":
            optimizer = torch.optim.AdamW(subnet.parameters(), lr=args.init_lr)
        if args.opt == "ranger":
            optimizer = Ranger(subnet.parameters(), lr=args.init_lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.T_max,
            eta_min=args.eta_min
        )

        optimal_epochs = train(
            args.epochs,
            subnet,
            data_provider,
            criterion,
            optimizer,
            scheduler,
            log_file,
            save_path,
            args.patience,
            step,
            branches_weights,
            ensemble_weights
        )

        # test
        test_loss, (test_top1, test_top5), _, _ = validate_ee(subnet, data_provider.test, criterion, branches_weights, ensemble_weights)
        test_log_str = f"TEST:  loss={test_loss:.3f}, top1={test_top1:.3f}, top5={test_top5:.3f}\n\n"
        log_file.write(test_log_str)

        # close log file
        log_file.close()

        args.epochs = optimal_epochs  # train with found number of epochs
        args.valid_size = None    # train on combined train and validation set

    yaml_path = os.path.join(save_path, 'args.yaml')
    with open(yaml_path, 'w') as f:
        f.write(args_text)
