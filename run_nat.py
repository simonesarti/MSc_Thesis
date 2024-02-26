# run neural architecture transfer
import argparse
import os

import numpy as np
import torch
import yaml

from NAT_extended.data_providers.providers_factory import get_data_provider
from NAT_extended.search.algorithms.nat import NAT
from NAT_extended.search.evaluators.ofa_evaluator import OFAEvaluator
from NAT_extended.search.search_spaces.search_spaces_factory import get_search_space
from NAT_extended.supernets.supernets_factory import get_supernet
from NAT_extended.train.trainer import SuperNetTrainer
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
        metavar='NAT DATASET',
        help='Name of the NAT dataset',
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
        default=None,  # if not set, use the one defined in the data_provider
        type=str,
        metavar='DATA',
        help='Path to the NAT dataset images'
    )
    parser.add_argument(
        '--n_exp',
        type=int,
        help='number of the current experiment on certain dataset+network',
        required=True
    )
    parser.add_argument(
        '--objs',
        default='acc+flops',
        type=str,
        metavar='OBJ',
        help='which objectives to optimize, separated by "+"',
        required=True
    )
    parser.add_argument(
        '--search_type',
        type=str,
        default="ea",
        choices=["ea"],
        help='search method to be used',
        required=True
    )
    parser.add_argument(
        '--save_path',
        default=None,  # if not set, use the default one below
        type=str,
        metavar='SAVE',
        help='path to dir for saving results'
    )
    parser.add_argument(
        '--resume_arch',
        default=None,
        type=str,
        metavar='RESUME_A',
        help='path to archive to resume the search'
    )
    parser.add_argument(
        '--resume_ckpt',
        default=None,
        type=str,
        metavar='RESUME_C',
        help='path to checkpoint to resume the search'
    )
    parser.add_argument(
        '--n_cores',
        default=1,
        type=int,
        help='number of cpus to be used'
    )

    args = parser.parse_args()

    # set save path
    if args.save_path is None:
        args.save_path = os.path.join(
            "running_experiment",
            "storage",
            "NAT",
            "NAT_TRAIN",
            "run",
            "from_%s" % args.ofa_dataset,
            "to_%s" % args.nat_dataset,
            args.network,
            "Nexp_%d" % args.n_exp,
        )

    if args.resume_arch is None or args.resume_ckpt is None:
        assert args.resume_arch is None and args.resume_ckpt is None

    args.objs = args.objs.replace("+", "&")     # used "+" instead of "&" in execution string otherwise CLI breaks

    assert args.n_cores > 0

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


def assign_search_arguments(args):
    args.surrogate = 'lgb'  # which surrogate model to fit accuracy predictor
    args.feature_encoding = 'one-hot'
    args.n_doe = 100
    args.n_gen = 8
    args.max_gens = 30
    args.topN = 150


def assign_adaptation_arguments(args):
    # supernet adaption related settings
    args.ft_epochs_gen = 5  # number of epochs to adapt supernet in each gen
    args.epochs = int(args.ft_epochs_gen * args.max_gens)  # number of epochs in total to adapt supernet
    args.lr = 2.5e-3  # initial learning rate
    args.lr_min = 0.0  # final learning rate
    args.momentum = 0.9  # optimizer momentum
    args.wd = 3e-4  # optimizer weight decay
    args.grad_clip = 5  # clip gradient norm
    args.model_ema = False
    args.dynamic_batch_size = 4
    args.kd_ratio = 1.0
    args.report_freq = 10

    args.patience = 8
    args.workers = 8
    args.n_max_val = 32     # set max to num of evaluated subnets ( set <0 to evaluate all)
    args.optimizer = "sgd"
    args.branches_weights = [0.34, 0.27, 0.20, 0.13, 0.06]
    args.ensemble_weights = [0.34, 0.27, 0.20, 0.13, 0.06]


parser = argparse.ArgumentParser(description='NAT Training')
args = parser_add_arguments(parser)

warmup_weights_path = os.path.join(
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
args.supernet_weights = os.path.join(warmup_weights_path, 'phase_3', 'model_best.pth.tar')
args.teacher_weights = os.path.join(warmup_weights_path, 'phase_2', 'model_best.pth.tar')

assign_dataset_arguments(args)
assign_search_arguments(args)
assign_adaptation_arguments(args)


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

    # construct the supernet
    supernet = get_supernet(
        net_name=args.network,
        n_classes=data_provider.n_classes,
        dropout_rate=0,
        search_space=search_space
    )

    # load pretrained supernet checkpoints weights
    supernet_state_dicts = torch.load(args.supernet_weights, map_location='cpu')
    state_dicts = [
        supernet_state_dicts['model_w1.0_state_dict'],
        supernet_state_dicts['model_w1.2_state_dict']
    ]
    supernet.load_state_dict(state_dicts)

    # push supernet to cuda
    supernet.cuda()

    # get the teacher model
    teacher_str = search_space.decode([np.array(search_space.ub)])[0]
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

    # define the supernet trainer
    trainer = SuperNetTrainer(
        supernet=supernet,
        teacher_model=teacher,
        search_space=search_space,
        n_epochs=args.epochs,
        lr_init=args.lr,
        data_provider=data_provider,
        cur_epoch=0,
        lr_end=args.lr_min,
        momentum=args.momentum,
        wd=args.wd,
        save_path=args.save_path,
        dynamic_batch_size=args.dynamic_batch_size,
        kd_ratio=args.kd_ratio,
        sub_train_size=data_provider.subset_size,
        sub_train_batch_size=data_provider.subset_batch_size,
        report_freq=args.report_freq,
        optim=args.optimizer,
        patience=args.patience,
        n_max_val=args.n_max_val,
        is_se="EE" not in args.network,
        branches_weights=args.branches_weights,
        ensemble_weights=args.ensemble_weights

    )

    # construct the evaluator
    evaluator = OFAEvaluator(
        supernet=supernet,
        data_provider=data_provider,
        sub_train_size=data_provider.subset_size,
        sub_train_batch_size=data_provider.subset_batch_size
    )

    # construct NAT search engine
    nas_method = NAT(
        search_space=search_space,
        evaluator=evaluator,
        trainer=trainer,
        objs=args.objs,
        surrogate=args.surrogate,
        n_doe=args.n_doe,
        n_gen=args.n_gen,
        max_gens=args.max_gens,
        topN=args.topN,
        ft_epochs_gen=args.ft_epochs_gen,
        save_path=args.save_path,
        resume_arch=args.resume_arch,
        resume_ckpt=args.resume_ckpt,
        n_cores=args.n_cores
    )

    # kick-off the search
    nas_method.search()

    yaml_path = os.path.join(args.save_path, 'args.yaml')
    with open(yaml_path, 'w') as f:
        f.write(args_text)

    return


if __name__ == '__main__':
    main()
