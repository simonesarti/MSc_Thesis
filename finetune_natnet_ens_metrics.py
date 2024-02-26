import argparse
import os
import json
from pathlib import Path
import numpy as np
import random
import yaml

import torch
from torchprofile import profile_macs

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
        "--subnets_path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--nw",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--search_type",
        type=str,
        default="ea"
    )

    args = parser.parse_args()
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

    args.train_batch_size = 64  # input batch size for training
    args.test_batch_size = 512  # input batch size for testing

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='finetune NATnets')
    args = parser_add_arguments(parser)

    work_dir = args.subnets_path
    #ft_subnet = ft_subnet[:-5]

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

    save_path = os.path.join(work_dir, "finetune")
    os.makedirs(save_path, exist_ok=True)

    # create and open log file
    log_file_path = os.path.join(save_path, "log_metrics_ens_103.txt")
    log_file = open(os.path.join(log_file_path), "w")

    for sub_number in [1,2,3,4]:
        subnet_file_name = f"subnet_{sub_number}.json"
        subnet_file_path = os.path.join(work_dir,subnet_file_name)


        if subnet_file_name in os.listdir(work_dir):
            print(f"Executing for subnet_{sub_number}")
            # convert net string into valid dictionary and extract the data
            with open(subnet_file_path, "r") as subnet_file:
                subnet_dict = json.load(subnet_file)
                subnet_arch = subnet_dict["arch"]
                resolution = subnet_arch.pop("r")

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

            # extract the subnet with the structure previously retrieved, and set the running statistics
            supernet.set_active_subnet(**subnet_arch)

            subnet = supernet.get_active_some_exits_subnet(list(range(1, subnet_arch["nd"]+1)), preserve_weight=True)
            branches_weights, ensemble_weights = get_weights(subnet_arch["nd"], "UNIF")

            sdl = data_provider.build_sub_train_loader(data_provider.subset_size, data_provider.subset_batch_size)
            set_running_statistics(subnet, sdl)

            # move network to gpu and define the loss calculation criterion
            subnet = subnet.cuda()

            params = sum(p.numel() for p in subnet.parameters() if p.requires_grad) / 1e6  # in unit of Million

            dummy_input_size = (1, 3, resolution, resolution)
            dummy_data = torch.rand(*dummy_input_size)
            dummy_data = dummy_data.cuda()
            macs = profile_macs(subnet, dummy_data) / 1e6  # in unit of MFLOPs

            # test
            test_log_str = f"subnet_{sub_number}: {params:.2f}\t{macs:.2f}\n"
            log_file.write(test_log_str)

        # close log file
    log_file.close()
