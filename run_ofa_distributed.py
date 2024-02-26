import argparse
import os
import random

import horovod.torch as hvd
import numpy as np
import torch
import torch.nn as nn
from ofa.imagenet_classification.elastic_nn.modules.dynamic_op import DynamicSeparableConv2d
from ofa.imagenet_classification.elastic_nn.training.progressive_shrinking import load_models

from OFA_mbv3_extended.my_data_loaders.my_random_resize_crop import MyModRandomResizedCrop
from OFA_mbv3_extended.networks.nets.my_networks import get_teacher_by_name, initialize_dyn_net
from OFA_mbv3_extended.run_manager import DistributedDatasetRunConfig, DistributedSingleExitRunManager, \
    DistributedEarlyExitRunManager
from OFA_mbv3_extended.train.utils import get_paths_dict, assign_phases_args, assign_dataset_args, assign_run_args

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="tiny_imagenet",
    choices=["tiny_imagenet", "imagenet"],
    required=True
)
parser.add_argument(
    "--network",
    type=str,
    default="OFAMobileNetV3",
    choices=[
        # ---- original elastic network ----
        "OFAMobileNetV3",
        # ---- new elastic nets SE ----
        "SE_B_OFAMobileNetV3",
        "SE_D_OFAMobileNetV3",
        "SE_P_OFAMobileNetV3",
        "SE_DP_OFAMobileNetV3",
        # ---- new elastic nets EE ----
        "EE_B_OFAMobileNetV3",
        "EE_D_OFAMobileNetV3",
        "EE_P_OFAMobileNetV3",
        "EE_DP_OFAMobileNetV3",
    ],
    required=True
)
parser.add_argument("--width_mult", type=float, default=1.0, choices=[1.0, 1.2], required=True)
parser.add_argument("--n_exp", type=int)
parser.add_argument(
    "--task",
    type=str,
    default="full",
    choices=["full", "kernel", "depth", "expand", "net_depth", "net_width"],
    required=True
)
parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3, 4])
parser.add_argument("--resume", action="store_true")

args = parser.parse_args()

assign_phases_args(args)

paths_dict = get_paths_dict(args)
args.path = paths_dict["current"]

assign_run_args(args)
assign_dataset_args(args)


if __name__ == "__main__":

    os.makedirs(args.path, exist_ok=True)

    # Initialize Horovod
    hvd.init()
    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(hvd.local_rank())

    num_gpus = hvd.size()

    # set seeds
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    # set image sizes
    args.image_size = [int(img_size) for img_size in args.image_size.split(",")]
    if len(args.image_size) == 1:
        args.image_size = args.image_size[0]
    MyModRandomResizedCrop.CONTINUOUS = args.continuous_size
    MyModRandomResizedCrop.SYNC_DISTRIBUTED = not args.not_sync_distributed_image_size

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        "momentum": args.momentum,
        "nesterov": not args.no_nesterov,
    }

    args.init_lr = args.base_lr * num_gpus  # linearly rescale the learning rate

    if args.warmup_lr < 0:
        args.warmup_lr = args.base_lr
    args.train_batch_size = args.base_batch_size
    args.test_batch_size = args.base_batch_size * 2

    run_config = DistributedDatasetRunConfig(*args.__dict__, num_replicas=num_gpus, rank=hvd.rank())

    # print run config information
    if hvd.rank() == 0:
        print("Run config:")
        for k, v in run_config.config.items():
            print("\t%s: %s" % (k, v))

    if args.dy_conv_scaling_mode == -1:
        args.dy_conv_scaling_mode = None
    DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = args.dy_conv_scaling_mode

    # define elastic values
    args.ks_list = [int(ks) for ks in args.ks_list.split(",")]
    args.expand_list = [int(e) for e in args.expand_list.split(",")]
    args.depth_list = [int(d) for d in args.depth_list.split(",")]
    args.net_depth_list = [int(nd) for nd in args.net_depth_list.split(",")]
    args.net_width_list = [int(nw) for nw in args.net_width_list.split(",")]

    # create the teacher network, max one
    teacher_net = get_teacher_by_name(args.network)(
        n_classes=run_config.data_provider.n_classes,
        bn_param=(args.bn_momentum, args.bn_eps),
        dropout_rate=0.2 if args.task == "full" else 0,
        width_mult=args.width_mult,
        ks=7,
        expand_ratio=6,
        depth_param=4
    )

    # if "full", the teacher is the network
    if args.task == "full":
        args.teacher_model = None
        net = teacher_net

    # otherwise create the dynamic network
    else:
        if args.kd_ratio > 0:
            args.teacher_model = teacher_net
            args.teacher_model.cuda()

        net = initialize_dyn_net(run_config.data_provider.n_classes, args)

    """ Distributed RunManager """
    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    
    if "EE" not in args.network:
        distributed_run_manager_class = DistributedSingleExitRunManager
    else:
        distributed_run_manager_class = DistributedEarlyExitRunManager

    distributed_run_manager = distributed_run_manager_class(
        path=args.path, 
        net=net, 
        run_config=run_config,
        hvd_compression=compression,
        backward_steps=args.dynamic_batch_size,
        is_root=(hvd.rank() == 0)
    )

    distributed_run_manager.save_config()
    # hvd broadcast
    distributed_run_manager.broadcast()

    # ==================================================================================================================

    # train
    if args.task == "full":
        distributed_run_manager.train(args)

    else:
        args.teacher_path = paths_dict["teacher"]

        # load teacher net weights
        if args.kd_ratio > 0:
            load_models(
                run_manager=distributed_run_manager,
                dynamic_net=args.teacher_model,
                model_path=args.teacher_path
            )

        image_size_list = [args.image_size] if isinstance(args.image_size, int) else [min(args.image_size), max(args.image_size)]
        validate_func_dict = {
            "image_size_list": image_size_list,
            "ks_list": sorted({min(args.ks_list), max(args.ks_list)}),
            "expand_ratio_list": sorted({min(args.expand_list), max(args.expand_list)}),
            "depth_list": sorted({min(net.depth_list), max(net.depth_list)}),
        }

        # add elastic net-depth/net-width if network supports them
        if "P" in args.network:
            validate_func_dict["net_width_list"] = sorted({min(net.net_width_list), max(net.net_width_list)})
        if "EE" in args.network:
            validate_func_dict["net_depth_list"] = sorted({min(net.net_depth_list), max(net.net_depth_list)})

        # progressive shrinking training
        from OFA_mbv3_extended.train.new_progressive_shrinking import train, validate

        # train elastic kernel
        if args.task == "kernel":

            validate_func_dict["ks_list"] = sorted(args.ks_list)
            if distributed_run_manager.start_epoch == 0:
                args.ofa_checkpoint_path = paths_dict["kernel"]

                dynamic_net = distributed_run_manager.net.module if isinstance(distributed_run_manager.net, nn.DataParallel) else distributed_run_manager.net
                load_models(distributed_run_manager, dynamic_net, args.ofa_checkpoint_path)

                (
                    subnet_losses_mean,
                    subnets_top1_mean,
                    subnets_top5_mean,
                    valid_log
                ) = validate(distributed_run_manager, is_test=True, **validate_func_dict)

                log_str = f"{subnet_losses_mean:.3f}\t"
                log_str += f"{subnets_top1_mean:.3f}\t"
                log_str += f"{subnets_top5_mean:.3f}\t"
                log_str += valid_log
                distributed_run_manager.write_log(log_str, "valid")

            else:
                raise NotImplementedError("OFA resume not implemented")

            train(
                distributed_run_manager,
                args,
                lambda _run_manager, epoch, is_test: validate(_run_manager, epoch, is_test, **validate_func_dict)
            )

        # train elastic net-width
        elif args.task == "net_width":

            if "P" not in args.network:
                raise ValueError("Elastic net-width is not supported for non-parallel networks")
            else:

                from OFA_mbv3_extended.train.new_progressive_shrinking import train_elastic_net_width

                if args.phase == 1:
                    args.ofa_checkpoint_path = paths_dict["net_width_1"]
                elif args.phase == 2:
                    args.ofa_checkpoint_path = paths_dict["net_width_2"]
                else:
                    raise NotImplementedError

                train_elastic_net_width(train, distributed_run_manager, args, validate_func_dict)

        # train elastic net-depth
        elif args.task == "net_depth":

            if "EE" not in args.network:
                raise ValueError("Elastic net-depth is not supported for single-exit networks")
            else:

                from OFA_mbv3_extended.train.new_progressive_shrinking import train_elastic_net_depth

                if args.phase == 1:
                    args.ofa_checkpoint_path = paths_dict["net_depth_1"]
                elif args.phase == 2:
                    args.ofa_checkpoint_path = paths_dict["net_depth_2"]
                elif args.phase == 3:
                    args.ofa_checkpoint_path = paths_dict["net_depth_3"]
                elif args.phase == 4:
                    args.ofa_checkpoint_path = paths_dict["net_depth_4"]
                else:
                    raise NotImplementedError

                train_elastic_net_depth(train, distributed_run_manager, args, validate_func_dict)

        # train elastic depth
        elif args.task == "depth":

            from OFA_mbv3_extended.train.new_progressive_shrinking import train_elastic_depth

            if args.phase == 1:
                args.ofa_checkpoint_path = paths_dict["depth_1"]
            elif args.phase == 2:
                args.ofa_checkpoint_path = paths_dict["depth_2"]
            else:
                raise NotImplementedError

            train_elastic_depth(train, distributed_run_manager, args, validate_func_dict)

        # train elastic expand
        elif args.task == "expand":

            from OFA_mbv3_extended.train.new_progressive_shrinking import train_elastic_expand

            if args.phase == 1:
                args.ofa_checkpoint_path = paths_dict["expand_1"]
            elif args.phase == 2:
                args.ofa_checkpoint_path = paths_dict["expand_2"]
            else:
                raise NotImplementedError

            train_elastic_expand(train, distributed_run_manager, args, validate_func_dict)

        else:
            raise NotImplementedError

    # ==================================================================================================================

    print("\n\nTESTING STEP\n\n")

    # testing
    run_config.n_elastic_val = 0  # test all combinations

    log_file_path = os.path.join(args.path, "logs", "test_console.txt")
    log_file = open(log_file_path, "w")

    if args.task == "full":

        if isinstance(distributed_run_manager, DistributedSingleExitRunManager):
            img_size, val_loss, val_acc, val_acc5 = distributed_run_manager.validate_all_resolution(is_test=True)

            test_log = "Test:\ttop1={:.3f}, top5={:.3f}\t".format(np.mean(val_acc), np.mean(val_acc5))
            for i_s, t_a in zip(img_size, val_acc):
                test_log += "(%d, %.3f), " % (i_s, t_a)
            log_file.write(test_log)

        else:
            (img_size_list, net_val_loss_list, net_val_top1_list, net_val_top5_list,
             branches_val_losses_list, branches_val_top1s_list, branches_val_top5s_list
             ) = distributed_run_manager.validate_all_resolution(is_test=True)

            test_log = "Test:\tNet top1={:.3f}, top5={:.3f}\t".format(np.mean(net_val_top1_list),
                                                                      np.mean(net_val_top5_list))
            for i_s, t_a in zip(img_size_list, net_val_top1_list):
                test_log += "(%d, %.3f), " % (i_s, t_a)

            test_log += "\nBranches top1: ["
            top1_mean = np.mean(branches_val_top1s_list, axis=0)
            for t1m in top1_mean:
                test_log += f"{t1m:.3f}, "
            test_log += "]\n"
            test_log += "Branches top5: ["
            top5_mean = np.mean(branches_val_top5s_list, axis=0)
            for t5m in top5_mean:
                test_log += f"{t5m:.3f}, "
            test_log += "]\n\n"

            log_file.write(test_log)

    else:

        # validation dictionary
        image_size_list = [args.image_size] if isinstance(args.image_size, int) else [min(args.image_size), max(args.image_size)]
        validate_func_dict = {
            "image_size_list": image_size_list,
            "ks_list": sorted({min(args.ks_list), max(args.ks_list)}),
            "expand_ratio_list": sorted({min(args.expand_list), max(args.expand_list)}),
            "depth_list": sorted({min(net.depth_list), max(net.depth_list)}),
        }
        # add elastic net-depth/net-width if network supports them
        if "P" in args.network:
            validate_func_dict["net_width_list"] = sorted({min(net.net_width_list), max(net.net_width_list)})
        if "EE" in args.network:
            validate_func_dict["net_depth_list"] = sorted({min(net.net_depth_list), max(net.net_depth_list)})

        # testing elastic phases
        if args.task == "kernel":
            validate_func_dict["ks_list"] = sorted(args.ks_list)

        elif args.task == "net_width":
            validate_func_dict["net_width_list"] = sorted(args.net_width_list)

        elif args.task == "net_depth":
            validate_func_dict["net_depth_list"] = sorted(args.net_depth_list)

        elif args.task == "depth":
            validate_func_dict["depth_list"] = sorted(args.depth_list)

        elif args.task == "expand":
            validate_func_dict["expand_ratio_list"] = sorted(args.expand_list)

        else:
            raise NotImplementedError

        # validate() writes in train_console.txt, add this not to mix training/validation and testing strings
        distributed_run_manager.write_log(log_str="\n\n\n\n\n======TESTING======\n\n", prefix="train", should_print=False)

        (subnets_losses_mean, subnets_top1_mean, subnets_top5_mean, valid_log) = validate(distributed_run_manager, is_test=True,
                                                                                          **validate_func_dict)

        test_log = f"Net top1={subnets_top1_mean:.3f}, top5={subnets_top5_mean:.3f}\n" + valid_log
        log_file.write(test_log)

    log_file.close()

    print("\n\nTEST DONE\n\n")


