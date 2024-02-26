import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from ofa.imagenet_classification.elastic_nn.modules.dynamic_op import DynamicSeparableConv2d

from OFA_mbv3_extended.networks.nets.my_networks import get_teacher_by_name, initialize_dyn_net
from OFA_mbv3_extended.train.utils import get_paths_dict, assign_phases_args, assign_dataset_args, assign_run_args
from OFA_mbv3_extended.run_manager import (
    DatasetRunConfig,
    SingleExitRunManager,
    EarlyExitRunManager,
)
from OFA_mbv3_extended.train.new_progressive_shrinking import validate

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="tiny_imagenet",
    choices=[
        "tiny_imagenet",
        "imagenet",
    ]
)
parser.add_argument(
    "--network",
    type=str,
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
    ]
)
parser.add_argument("--width_mult", type=float, default=1.0, choices=[1.0, 1.2])
parser.add_argument("--n_exp", type=int)
parser.add_argument(
    "--task",
    type=str,
    default="full",
    choices=[
        "full",
        "kernel",
        "depth",
        "expand",
        "net_depth",
        "net_width"
    ]
)
parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3, 4], required=False)
args = parser.parse_args()

assign_phases_args(args)

paths_dict = get_paths_dict(args)
args.path = paths_dict["current"]

assign_run_args(args)
args.n_elastic_val = 0  # test all combinations

assign_dataset_args(args)

#=======================================================================================================================

# set image sizes
args.image_size = [int(img_size) for img_size in args.image_size.split(",")]
if len(args.image_size) == 1:
    args.image_size = args.image_size[0]

# define the run config and data provider (here we care about the test batch size)
args.train_batch_size = 100
args.test_batch_size = 100

run_config = DatasetRunConfig(**args.__dict__)

if args.dy_conv_scaling_mode == -1:
    args.dy_conv_scaling_mode = None
DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = args.dy_conv_scaling_mode

# define elastic values
args.ks_list = [int(ks) for ks in args.ks_list.split(",")]
args.expand_list = [int(e) for e in args.expand_list.split(",")]
args.depth_list = [int(d) for d in args.depth_list.split(",")]
args.net_depth_list = [int(nd) for nd in args.net_depth_list.split(",")]
args.net_width_list = [int(nw) for nw in args.net_width_list.split(",")]


# create the network to test
if args.task == "full":
    raise ValueError("full non serve")
else:
    net = initialize_dyn_net(run_config.data_provider.n_classes, args)

# choose run manager class
if "EE" not in args.network:
    run_manager_class = SingleExitRunManager
else:
    run_manager_class = EarlyExitRunManager

# create run manager and avoid file overwriting
os.rename(os.path.join(args.path, "net_info.txt"), os.path.join(args.path, "net_info_original.txt"))
run_manager = run_manager_class(path=args.path, net=net, run_config=run_config)
os.remove(os.path.join(args.path, "net_info.txt"))
os.rename(os.path.join(args.path, "net_info_original.txt"), os.path.join(args.path, "net_info.txt"))


# load checkpoint
net = run_manager.net.module if isinstance(run_manager.net, nn.DataParallel) else run_manager.net
checkpoint_path = os.path.join(args.path, "checkpoint", "model_best.pth.tar")
state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
net.load_state_dict(state_dict)

# setup logging
if args.task == "kernel":
    f_path = Path(args.path).resolve().parents[3]
else:
    f_path = Path(args.path).resolve().parents[4]
print(f_path)
log_file_path1 = os.path.join(f_path, "all_tests.txt")
log_file_path2 = os.path.join(f_path, "max_all_tests.txt")
log_file1 = open(log_file_path1, "a")
log_file2 = open(log_file_path2, "a")



# testing
if args.task != "full":

    # validation dictionary
    validate_func_dict = {
        "image_size_list": {args.image_size} if isinstance(args.image_size, int) else sorted(args.image_size),
        "ks_list": sorted(args.ks_list),
        "expand_ratio_list": sorted(args.expand_list),
        "depth_list": sorted(args.depth_list),
    }
    # add elastic net-depth/net-width if network supports them
    if "P" in args.network:
        validate_func_dict["net_width_list"] = sorted(args.net_width_list)
    if "EE" in args.network:
        validate_func_dict["net_depth_list"] = sorted(args.net_depth_list)

    # validate() writes in train_console.txt, add this not to mix training/validation and testing strings
    run_manager.write_log(log_str="\n\n\n\n\n======TESTING======\n\n", prefix="train", should_print=False)

    (subnets_losses_mean, subnets_top1_mean, subnets_top5_mean, valid_log) = validate(run_manager, is_test=True, **validate_func_dict)
    test_log = f"{args.network}-{args.width_mult}-{args.n_exp}-{args.task}-Net top1={subnets_top1_mean:.3f}, top5={subnets_top5_mean:.3f}\n" + valid_log + "\n\n"
    log_file1.write(test_log)

    valid_log = valid_log.split(" ")
    valid_arch = [valid_log[i] for i in range(0,len(valid_log),2)]
    valid_perf = [valid_log[i] for i in range(1,len(valid_log),2)]
    values = [float(elem.replace("(", "").replace(")", "").replace(",", "")) for elem in valid_perf]
    best_sub = max(values)
    best_idx = values.index(best_sub)
    best_arch = valid_arch[best_idx]

    test_log2 = f"{args.network}-{args.width_mult}-{args.n_exp}-{args.task}\n{subnets_top1_mean:.3f}\t{best_sub:.3f}\t{best_arch}\n"
    test_log2 = test_log2.replace(".", ",")
    log_file2.write(test_log2)

log_file1.close()
log_file2.close()