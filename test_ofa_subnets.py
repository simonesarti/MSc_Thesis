import argparse
import json
import os
import random
import time

import numpy as np
import torch
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics
from ofa.utils import AverageMeter
from torch import nn
from tqdm import tqdm

from NAT_extended.evaluation.utils import get_metric_dict, update_metric, get_metric_vals
from OFA_mbv3_extended.data_providers.imagenet import ImagenetDataProvider
from OFA_mbv3_extended.data_providers.tiny_imagenet import TinyImagenetDataProvider
from OFA_mbv3_extended.networks.nets.my_networks import initialize_dyn_net


def read_arguments(parser):

    parser.add_argument(
        "--dataset",
        type=str,
        default="tiny_imagenet",
        choices=["tiny_imagenet", "imagenet"],
        required=True
    )

    parser.add_argument(
        "--network",
        default="OFAMobileNetV3",
        choices=[
            # original elastic network -------------
            "OFAMobileNetV3",
            # new elastic nets ----------
            "SE_B_OFAMobileNetV3",
            "SE_D_OFAMobileNetV3",
            "SE_P_OFAMobileNetV3",
            "SE_DP_OFAMobileNetV3",
            "EE_B_OFAMobileNetV3",
            "EE_D_OFAMobileNetV3",
            "EE_P_OFAMobileNetV3",
            "EE_DP_OFAMobileNetV3",
        ],
        required=True
    )
    parser.add_argument("--width_mult", type=float, default=1.0, choices=[1.0, 1.2], required=True)
    parser.add_argument("--n_exp", type=int, required=True)
    parser.add_argument(
        "--test_mode",
        type=str,
        choices=[
            "randomN",      # test N randomly sampled networks
            "custom_nets"     # test the architectures specified in a file
        ],
        required=True
    )

    parser.add_argument("--workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--continuous_size", type=bool, default=False, help="if True, image sizes vary in steps of 4")

    parser.add_argument("--n_random", type=int, default=0, help="number of random networks to test")
    # test architectures specified in file, expected list of dictionaries {"r":N, "ks":[...], ....}
    parser.add_argument("--conf_file", type=str, help="path of file containing the architectures to test")

    args = parser.parse_args()

    # check that the arguments for the test modes have been passed
    if args.test_mode == "randomN":
        assert args.n_random > 0, "A number of architectures >0 must be specified for randomN test mode"
    if args.test_mode == "custom_nets":
        assert args.conf_file is not None, "A valid path must be specified for custom_nets test mode"
        assert os.path.exists(args.conf_file), "The path must exits"
        assert os.path.isfile(args.conf_file), "The path must refer to a file (.JSON)"

    # dataset related settings
    if args.dataset == "imagenet":
        args.image_size = [size for size in range(128, 225, 4)] if args.continuous_size else [128, 160, 192, 224]
        args.resize_scale = 0.08
    elif args.dataset == "tiny_imagenet":
        args.image_size = [size for size in range(48, 65, 4)] if args.continuous_size else [48, 56, 64]
        args.resize_scale = 0.85
    else:
        raise NotImplementedError

    return args


def log_string(log_file, log_str):
    with open(log_file, "a") as logger:
        logger.write(log_str)


def reset_running_statistics(net, data_provider):
    subset_size = data_provider.subset_size
    subset_batch_size = data_provider.subset_batch_size
    data_loader = data_provider.build_sub_train_loader(subset_size, subset_batch_size)
    set_running_statistics(net, data_loader)


def test_net(net, data_loader, criterion, device, n):

    net.eval()

    losses = AverageMeter()
    metric_dict = get_metric_dict()

    with torch.no_grad():
        with tqdm(
                total=len(data_loader),
                desc='Testing subnet #{}'.format(n)
        ) as t:

            for batch, (images, labels) in enumerate(data_loader):
                images, labels = images.to(device), labels.to(device)

                # compute output
                output = net(images)
                loss = criterion(output, labels)

                # measure accuracy and record loss
                losses.update(loss.item(), images.size(0))
                update_metric(metric_dict, output, labels)

                t.set_postfix({
                    'loss': losses.avg,
                    **get_metric_vals(metric_dict, return_dict=True),
                    'img_size': images.size(2),
                })
                t.update(1)

    return losses.avg, get_metric_vals(metric_dict)


def run_and_log(net, test_loader, log_file, log_str, n):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    loss, (top1, top5) = test_net(net, test_loader, criterion, device, n)
    err1, err5 = 100-top1, 100-top5

    to_log = f"NET #{n}: " + log_str + "\n"
    to_log += f"RESULTS: loss={loss:.3f},\t top1={top1:.3f} (err1={err1:.3f}),\t top5={top5:.3f} (err5={err5:.3f})\n"
    to_log += "MODULE STRING:\n" + net.module_str + "\n--------------------------------\n\n"
    log_string(log_file, to_log)

    return loss, top1, top5


parser = argparse.ArgumentParser()
args = read_arguments(parser)

# non distributed, single gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if args.dataset == "tiny_imagenet":
    DataProviderClass = TinyImagenetDataProvider
if args.dataset == "imagenet":
    DataProviderClass = ImagenetDataProvider

data_provider = args.DataProviderClass(
        train_batch_size=1,
        test_batch_size=64,
        valid_size=None,
        n_worker=args.workers,
        image_size=args.image_size,
        resize_scale=args.resize_scale,
        distort_color=None,
        num_replicas=None,
        rank=None
)

n_classes = data_provider.n_classes

# default values for networks and create it
args.bn_momentum = 0.1
args.bn_eps = 1e-5
args.dropout = 0
args.ks_list = [3, 5, 7]
args.expand_list = [3, 4, 6]
args.depth_list = [2, 3, 4]
args.net_depth_list = [1, 2, 3, 4, 5]
args.net_width_list = [1, 2, 3]
ofa_network = initialize_dyn_net(n_classes, args)

base_path = os.path.join(
        "running_experiment",
        "storage",
        "OFA",
        "train",
        args.dataset,
        args.network,
        "wm_" + str(args.width_mult),
        "Nexp_" + str(args.n_exp)
    )

# load network weights
model_path = os.path.join(base_path, "expand", "phase_2", "checkpoint", "model_best.pth.tar")
state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
ofa_network.load_state_dict(state_dict)

# create log files and run tests
log_folder = os.path.join(base_path, "evaluation")
os.makedirs(log_folder, exist_ok=True)
log_file_name = args.test_mode

losses = []
top1s = []
top5s = []

if args.test_mode == "randomN":
    timestr = time.strftime("%Y%m%d-%H%M%S")
    log_file_name += f"@{args.n_random}_" + timestr + ".txt"
    log_file = os.path.join(log_folder, log_file_name)
    resolutions = [random.choice(args.image_size) for _ in range(args.n_random)]
    for i in range(args.n_random):
        setting = ofa_network.sample_active_subnet()
        subnet = ofa_network.get_active_subnet(preserve_weight=True)
        data_provider.assign_active_img_size(resolutions[i])
        reset_running_statistics(subnet, data_provider)
        setting["r"] = resolutions[i]
        log_str = json.dumps(setting)
        loss, top1, top5 = run_and_log(subnet, data_provider.test, log_file, log_str, i+1)
        losses.append(loss)
        top1s.append(top1)
        top5s.append(top5)

"""
json file structure should be: (an array of dicts)

[
    {"r":48, "ks":[3,7,5,7, ...], "d": [3,4,2,4,3], "e":[3,6,4, ...]},
    {"r":48, "ks":7, "d": 4, "e":[3,6,4, ...], ....},
    ...
    {"r":48, "ks":[3,7,5,7, ...], "d": 4, "e":3, ....}
]

r, ks, e, d always present
nd, nw presence depends on type of network

"""
if args.test_mode == "custom_nets":
    timestr = time.strftime("%Y%m%d-%H%M%S")
    log_file_name += "_" + timestr + ".txt"
    log_file = os.path.join(log_folder, log_file_name)
    with open(args.conf_file, 'r') as jf:
        data = json.load(jf)
        arch_list = []
        for arch in data:
            arch_list.append(arch)
    for i, arch in enumerate(arch_list, 1):
        log_str = json.dumps(arch)
        resolution = arch.pop("r")
        ofa_network.set_active_subnet(**arch)
        subnet = ofa_network.get_active_subnet(preserve_weight=True)
        data_provider.assign_active_img_size(resolution)
        reset_running_statistics(subnet, data_provider)
        loss, top1, top5 = run_and_log(subnet, data_provider.test, log_file, log_str, i)
        losses.append(loss)
        top1s.append(top1)
        top5s.append(top5)


# log results
losses_arr = np.array(losses)
top1s_arr = np.array(top1s)
top5s_arr = np.array(top5s)
mean_loss, min_loss, idx_min_loss = np.mean(losses_arr), np.min(losses_arr), np.argmin(losses_arr) + 1
mean_top1, max_top1, idx_max_top1 = np.mean(top1s_arr), np.max(top1s_arr), np.argmax(top1s_arr) + 1
mean_top5, max_top5, idx_max_top5 = np.mean(top5s_arr), np.max(top5s_arr), np.argmax(top5s_arr) + 1

to_log = "\n\nREPORT:\n\n"
to_log += f"mean loss:{mean_loss:.3f}\n"
to_log += f"mean top1:{mean_top1:.3f} (err1:{(100-mean_top1):.3f})\n"
to_log += f"mean top5:{mean_top5:.3f} (err5:{(100-mean_top5):.3f})\n\n"
to_log += f"Best subnet loss: network #{idx_min_loss} with loss={min_loss:.3f}\n"
to_log += f"Best subnet top1: network #{idx_max_top1} with top1={max_top1:.3f} (err1={(100-max_top1):.3f})\n"
to_log += f"Best subnet top5: network #{idx_max_top5} with top5={max_top5:.3f} (err5={(100-max_top5):.3f})\n"

log_string(log_file, to_log)

















