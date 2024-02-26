# before running NSGANetV2 or NAT, run this script first to ensure better performance
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from NAT_extended.data_providers.providers_factory import get_data_provider
from NAT_extended.search.search_spaces.search_spaces_factory import get_search_space
from NAT_extended.supernets.supernets_factory import get_supernet
from NAT_extended.supernets.utils import reset_classifier
from NAT_extended.train.trainer import TrainerSE, TrainerEE, SuperNetTrainer
from OFA_mbv3_extended.my_data_loaders.my_random_resize_crop import MyModRandomResizedCrop


def parser_add_arguments(parser):
    parser.add_argument(
        "--network",
        type=str,
        default="OFAMobileNetV3",
        required=True,
        choices=[
            # original elastic network -------------
            "OFAMobileNetV3",
            # new elastic nets, Single-exit ----------
            "SE_B_OFAMobileNetV3",
            "SE_D_OFAMobileNetV3",
            "SE_P_OFAMobileNetV3",
            "SE_DP_OFAMobileNetV3",
            # new elastic nets, Early-Exit ----------
            "EE_B_OFAMobileNetV3",
            "EE_D_OFAMobileNetV3",
            "EE_P_OFAMobileNetV3",
            "EE_DP_OFAMobileNetV3",
        ],
    )
    parser.add_argument(
        "--ofa_dataset",
        type=str,
        default="tiny_imagenet",
        choices=["tiny_imagenet", "imagenet"],
        required=True
    )
    parser.add_argument(
        '--nat_dataset',
        default='cifar10',
        type=str,
        required=True,
        metavar='DATASET',
        help='Name of the transfer dataset',
        choices=[
            "tiny_imagenet",
            "imagenet",
            "cifar10",
            "cifar100",
            "cinic10",
            "aircraft",
            "cars",
            "dtd",
            "flowers102",
            "food101",
            "pets",
            "stl10"
        ]
    )
    parser.add_argument(
        '--nat_dataset_path',
        default=None,   # if not set, use the one defined in the data_provider
        type=str,
        metavar='DATA',
        help='Path to the NAT dataset images'
    )
    parser.add_argument(
        '--phase',
        type=int,
        default=1,
        choices=[1, 2, 3],
        metavar='P',
        help='which training phase to run (default: 1)',
        required=True,
    )
    parser.add_argument(
        '--search_type',
        type=str,
        default="ea",
        choices=["ea"],
        help='search method to be used',
        required=True,
    )
    parser.add_argument(
        '--save_path',
        default=None,   # if not set, use the default one below
        type=str,
        metavar='SAVE',
        help='path to dir for saving results'
    )

    args = parser.parse_args()

    # set save path
    if args.save_path is None:
        args.save_path = os.path.join(
            "running_experiment",
            "storage",
            "NAT",
            "NAT_TRAIN",
            "warmup",
            "from_%s" % args.ofa_dataset,
            "to_%s" % args.nat_dataset,
            args.network,
            "supernet"
        )

    return args


def assign_dataset_arguments(args):

    if args.ofa_dataset == "tiny_imagenet":
        args.image_sizes = [48, 56, 64]
    if args.ofa_dataset == "imagenet":
        args.image_sizes = [128, 160, 192, 224]

    args.train_batch_size = 256  # input batch size for training    # change for parallel networks
    args.test_batch_size = 512  # input batch size for testing      # change for parallel networks

    if args.nat_dataset == "tiny_imagenet":
        args.valid_size = 0.15
        args.resize_scale = 0.85
    elif args.nat_dataset == "imagenet":
        args.valid_size = 100000
        args.resize_scale = 0.08
    elif args.nat_dataset == "cifar10":
        args.valid_size = 5000
        args.resize_scale = 1.0
    elif args.nat_dataset == "cifar100":
        args.valid_size = 5000
        args.resize_scale = 1.0
    elif args.nat_dataset == "cinic10":
        args.valid_size = None  # use default val dataset
        args.resize_scale = 1.0
    elif args.nat_dataset == "aircraft":
        args.valid_size = None  # use default val dataset
        args.resize_scale = 0.35
    elif args.nat_dataset == "cars":
        args.valid_size = 0.15
        args.resize_scale = 0.25
    elif args.nat_dataset == "dtd":
        args.valid_size = None  # use default val dataset
        args.resize_scale = 0.2
    elif args.nat_dataset == "flowers102":
        args.valid_size = 1000
        args.resize_scale = 1.0
    elif args.nat_dataset == "food101":
        args.valid_size = 0.15
        args.resize_scale = 1.0
    elif args.nat_dataset == "pets":
        args.valid_size = 0.15
        args.resize_scale = 1.0
    elif args.nat_dataset == "stl10":
        args.valid_size = 0.15
        args.resize_scale = 0.75
    else:
        raise NotImplementedError


def assign_training_arguments(args):
    args.workers = 8
    args.branches_weights = [0.34, 0.27, 0.20, 0.13, 0.06]
    args.ensemble_weights = [0.34, 0.27, 0.20, 0.13, 0.06]
    args.patience = 8
    args.n_max_val = 32     # set max to num of evaluated subnets ( set <0 to evaluate all)
    args.optimizer_name = "sgd"

    args.feature_encoding = 'one-hot'

    # training related settings
    if args.phase < 3:  # 1 or 2
        args.epochs = 100  # number of epochs to train
        args.lr = 7.5e-3  # initial learning rate
        args.lr_min = 0.0  # final learning rate
        args.lr_warmup_epochs = 5  # number of epochs to warm-up learning rate
        args.momentum = 0.9  # optimizer momentum
        args.wd = 3e-4  # optimizer weight decay
        args.grad_clip = 5  # clip gradient norm
        args.model_ema = False
        args.save_path = os.path.join(args.save_path, "phase_%d" % args.phase)    # phase_1: wm=1.0, phase_2: wm=1.2

    else:
        args.epochs = 150  # number of epochs to train
        args.lr = 7.5e-3  # 0.025  # initial learning rate
        args.lr_min = 0.0  # final learning rate
        args.lr_warmup_epochs = 10  # number of epochs to warm-up learning rate
        args.momentum = 0.9  # optimizer momentum
        args.wd = 3e-4  # optimizer weight decay
        args.grad_clip = 5  # clip gradient norm
        args.model_ema = False
        args.dynamic_batch_size = 4
        args.kd_ratio = 1.0
        args.report_freq = 10
        args.teacher_weights = os.path.join(args.save_path, "phase_2", "model_best.pth.tar")
        args.save_path = os.path.join(args.save_path, "phase_3")
        # use as teacher weights the ones for width-multiplier 1.2


parser = argparse.ArgumentParser(description='Warm-up Supernet Training')
args = parser_add_arguments(parser)
assign_dataset_arguments(args)
assign_training_arguments(args)


def main():
    os.makedirs(args.save_path, exist_ok=True)

    # Cache the args as a text string to save them in the output dir
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    MyModRandomResizedCrop.CONTINUOUS = True
    MyModRandomResizedCrop.SYNC_DISTRIBUTED = True

    # construct the data provider
    data_provider = get_data_provider(
        dataset=args.nat_dataset,
        save_path=args.nat_dataset_path,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        valid_size=args.valid_size,
        n_worker=args.workers,
        image_size=args.image_sizes,
        resize_scale=args.resize_scale,
        distort_color=None,
        num_replicas=None,
        rank=None
    )

    # construct the search space
    search_space = get_search_space(
        net_name=args.network,
        search_type=args.search_type,
        image_scale_list=args.image_sizes,
        feature_encoding=args.feature_encoding
    )

    # construct the supernet (assuming pretrained)
    supernet = get_supernet(
        net_name=args.network,
        n_classes=200 if args.phase < 3 else data_provider.n_classes,
        dropout_rate=0,
        search_space=search_space
    )
    # n_classes= 200 in phases 1&2 to be able to load weights from tiny_imagenet, after loading reset the classifier
    # and proceed with the training on the new dataset

    if args.phase < 3:  # 1 or 2
        # use the results from OFA training on tiny_imagenet dataset
        common_path = os.path.join("running_experiment", "storage", "OFA", "results", args.ofa_dataset, args.network)
        path_w1 = os.path.join(common_path, "wm_10.pth.tar")
        path_w2 = os.path.join(common_path, "wm_12.pth.tar")


        state_dicts = [
            torch.load(path_w1, map_location='cpu')['state_dict'],
            torch.load(path_w2, map_location='cpu')['state_dict']
        ]

        supernet.load_state_dict(state_dicts)

        # change the task-specific layer accordingly
        for model in supernet.engine:
            if args.network == "OFAMobileNetV3":    # original network
                reset_classifier(model, n_classes=data_provider.n_classes)
            else:   # staged networks
                if "EE" in args.network:
                    for exit_stage in model.exit_stages:
                        exit_stage.reset_classifier(n_classes=data_provider.n_classes)
                else:
                    model.exit_stage.reset_classifier(n_classes=data_provider.n_classes)

        # pretrained the full capacity supernet as teacher
        arch = search_space.decode([np.array(search_space.ub)])[0]  # [0] because decode() returns list
        if args.phase < 2:
            arch['w'] = 1.0     # in phase 2 "w" already set correctly to 1.2 due to decoding of upper bound
        print(arch)

        # set the teacher architecture (complete configuration)
        supernet.set_active_subnet(**arch)
        if "EE" not in args.network:
            teacher = supernet.get_active_subnet(preserve_weight=True)
            # get the static single-exit subnet given the active structure of the net
        else:
            teacher = supernet.get_active_all_exits_subnet(preserve_weight=True)
            # get the static multi-exit subnet, as teacher EE should have all the exits
        teacher = teacher.cuda()

        # define the trainer
        trainer_arguments = {
            "model": teacher,
            "n_epochs": args.epochs,
            "lr_init": args.lr,
            "data_provider": data_provider,
            "optimizer": args.optimizer_name,
            "patience": args.patience,
            "cur_epoch": 0,
            "lr_end": args.lr_min,
            "lr_warmup_epochs": args.lr_warmup_epochs,
            "momentum": args.momentum,
            "wd": args.wd,
            "save_path": args.save_path,
        }
        if "EE" in args.network:
            trainer_arguments["branches_weights"] = args.branches_weights
            trainer_arguments["ensemble_weights"] = args.ensemble_weights
            trainer = TrainerEE(**trainer_arguments)
            trainer.train()     # kick-off the training

            # testing
            trainer.logger.info("\n\n\nTEST averages:\n")
            t_net_loss, (t_net_top1, t_net_top5), t_br_loss, (t_br_top1, t_br_top5) = trainer.validate(epoch=0, is_test=True)
            t_br_loss_str = "[" + ", ".join([str(s) for s in [round(elem, 4) for elem in t_br_loss]]) + "]"
            t_br_top1_str = "[" + ", ".join([str(s) for s in [round(elem, 4) for elem in t_br_top1]]) + "]"
            t_br_top5_str = "[" + ", ".join([str(s) for s in [round(elem, 4) for elem in t_br_top5]]) + "]"

            test_str = "\nnet_loss={:.4f}, net_top1={:.4f}, net top5={:.4f} ".format(t_net_loss, t_net_top1, t_net_top5)
            test_str += "\nbr_loss=" + t_br_loss_str + "\nbr_top1=" + t_br_top1_str + "\nbr_top5=" + t_br_top5_str
            trainer.logger.info(test_str)

        else:
            trainer = TrainerSE(**trainer_arguments)
            trainer.train()  # kick-off the training

            # testing
            trainer.logger.info("\n\n\nTEST averages:\n")
            test_loss, (test_acc1, test_acc5) = trainer.validate(epoch=0, is_test=True)
            test_str = "loss={:.4f}, top1={:.4f}, top5={:.4f}".format(test_loss, test_acc1, test_acc5)
            trainer.logger.info(test_str)

    else:

        # load pretrained checkpoints weights from previous phases
        path_w1 = os.path.join(Path(args.save_path).parent, "phase_1", "model_best.pth.tar")
        path_w2 = os.path.join(Path(args.save_path).parent, "phase_2", "model_best.pth.tar")

        state_dicts = [
            torch.load(path_w1, map_location='cpu')['model_state_dict'],
            torch.load(path_w2, map_location='cpu')['model_state_dict']
        ]

        supernet.load_state_dict(state_dicts)
        for engine in supernet.engine:
            engine.re_organize_middle_weights(expand_ratio_stage=1)

        # warm-up sub-parts of supernet by uniform sampling
        # push supernet to cuda
        supernet.cuda()

        # get the teacher model
        teacher_str = search_space.decode([np.array(search_space.ub)])[0]   # [0] because decode() returns list
        supernet.set_active_subnet(**teacher_str)

        if "EE" not in args.network:
            teacher = supernet.get_active_subnet(preserve_weight=False)
            # get the static single-exit subnet given the active structure of the net
        else:
            teacher = supernet.get_active_all_exits_subnet(preserve_weight=False)
            # get the static multi-exit subnet, as teacher EE should have all the exits

        teacher_state_dict = torch.load(args.teacher_weights, map_location='cpu')['model_state_dict']
        teacher.load_state_dict(teacher_state_dict)
        teacher = teacher.cuda()

        distributions = None
        # # construct the distribution, just for debug
        # import json
        # from NAT_extended.search.algorithms.evo_nas import EvoNAS
        # from NAT_extended.search.algorithms.utils import distribution_estimation, rank_n_crowding_survival
        #
        # archive_path = os.path.join(
        #         #     pathlib.Path(__file__).parent.resolve(),
        #         #     "tmp/MobileNetV3SearchSpaceNSGANetV2-acc&flops-lgb-n_doe@100-n_iter@8-max_iter@30/"
        #         #     "iter_30/archive.json"
        # archive = json.load(open(archive_path), 'r'))
        # archs = [m['arch'] for m in archive]
        # X = search_space.encode(archs)
        # F = np.array(EvoNAS.get_attributes(archive, _attr='err&flops'))
        #
        # print(X.shape)
        # print(F.shape)
        #
        # sur_X = rank_n_crowding_survival(X, F, n_survive=150)
        # distributions = []
        # for j in range(X.shape[1]):
        #     distributions.append(distribution_estimation(sur_X[:, j]))

        # define the trainer
        trainer = SuperNetTrainer(
            supernet=supernet,
            teacher_model=teacher,
            search_space=search_space,
            n_epochs=args.epochs,
            lr_init=args.lr,
            data_provider=data_provider,
            cur_epoch=0,
            lr_end=args.lr_min,
            lr_warmup_epochs=args.lr_warmup_epochs,
            momentum=args.momentum,
            wd=args.wd,
            save_path=args.save_path,
            dynamic_batch_size=args.dynamic_batch_size,
            kd_ratio=args.kd_ratio,
            sub_train_size=data_provider.subset_size,
            sub_train_batch_size=data_provider.subset_batch_size,
            distributions=distributions,
            report_freq=args.report_freq,
            optim=args.optimizer_name,
            patience=args.patience,
            n_max_val=args.n_max_val,
            is_se="EE" not in args.network,
            branches_weights=args.branches_weights,
            ensemble_weights=args.ensemble_weights
        )

        # kick-off the training
        trainer.train()

        # testing
        trainer.n_max_val = 0   # test all combinations of network extremes
        trainer.logger.info("\n\n\nTEST:\n\n")
        test_loss, (test_acc1, test_acc5) = trainer.validate(epoch=0, is_test=True)
        test_str = "\n\nTEST: loss_avg={:.4f}, top1_avg={:.4f}, top5_avg={:.4f}".format(test_loss, test_acc1, test_acc5)
        trainer.logger.info(test_str)

    yaml_path = os.path.join(args.save_path, 'args.yaml')
    with open(yaml_path, 'w') as f:
        f.write(args_text)

    return


if __name__ == '__main__':
    main()
