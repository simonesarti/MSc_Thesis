import argparse
import json
import os

import torch
from timm.utils import accuracy, AverageMeter
from tqdm import tqdm
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics
from NAT_extended.search.search_spaces.search_spaces_factory import get_search_space
from NAT_extended.supernets.supernets_factory import get_supernet
from NAT_extended.data_providers.providers_factory import get_data_provider

"""not needed anymore, incorporated and expanded directly after search"""


def assign_dataset_arguments(args):
    # assuming pretrained on tiny imagenet (sizes 48->64)
    args.train_batch_size = 100  # input batch size for training #change for Parallel networks
    args.test_batch_size = 100  # input batch size for testing  #change for Parallel networks
    args.image_sizes = [48, 56, 64]

    if args.dataset == "tiny_imagenet":
        args.valid_size = 0.15
        args.resize_scale = 0.85
    elif args.dataset == "imagenet":
        args.valid_size = 100000
        args.resize_scale = 0.08
    elif args.dataset == "cifar10":
        args.valid_size = 5000  # use additional dataset
        args.resize_scale = 1.0
    elif args.dataset == "cifar100":
        args.valid_size = 5000
        args.resize_scale = 1.0
    elif args.dataset == "cinic10":
        args.valid_size = None  # use default val dataset
        args.resize_scale = 1.0
    elif args.dataset == "aircraft":
        args.valid_size = None  # use default val dataset
        args.resize_scale = 0.15
    elif args.dataset == "cars":
        args.valid_size = 0.15
        args.resize_scale = 0.15
    elif args.dataset == "dtd":
        args.valid_size = None  # use default val dataset
        args.resize_scale = 0.15
    elif args.dataset == "flowers102":
        args.valid_size = 0.5
        args.resize_scale = 1.0
    elif args.dataset == "food101":
        args.valid_size = 5000
        args.resize_scale = 1.0
    elif args.dataset == "pets":
        args.valid_size = 0.15
        args.resize_scale = 1.0
    elif args.dataset == "stl10":
        args.valid_size = 0.15
        args.resize_scale = 0.75
    else:
        raise NotImplementedError


def validate(model, loader, criterion, net_name):

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    with torch.no_grad():
        with tqdm(total=len(loader), desc="Testing " + net_name) as t:

            for i, (input, target) in enumerate(loader):
                target = target.cuda()
                input = input.cuda()

                # compute output
                output = model(input)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1.item(), input.size(0))
                top5.update(acc5.item(), input.size(0))

                t.update(1)

        return losses.avg, top1.avg, top5.avg


def main(args):

    # get the dataloaders for the dataset, depending on image sizes
    data_provider = get_data_provider(
        dataset=args.dataset,
        save_path=args.dataset_path,
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

    # build the search space for the supernet
    search_space = get_search_space(
        net_name=args.supernet,
        search_type=args.search_type,
        image_scale_list=args.image_sizes
    )
    # construct the supernet given the search space
    supernet = get_supernet(
        net_name=args.supernet,
        n_classes=data_provider.n_classes,  # extract number of classes from one of the dataloaders
        dropout_rate=0,
        search_space=search_space
    )
    # load the weights from the checkpoint
    supernet_state_dicts = torch.load(args.weights_path, map_location='cpu')
    state_dicts = [
        supernet_state_dicts['model_w1.0_state_dict'],
        supernet_state_dicts['model_w1.2_state_dict']
    ]
    supernet.load_state_dict(state_dicts)

    # create logging file
    log_file_path = os.path.join(args.subnets_folder, "test_log.txt")
    module_strings_path = os.path.join(args.subnets_folder, "module_strings.txt")

    if os.path.exists(log_file_path):
        print("already tested, check test_log.txt file in the specified folder")
        return

    log_file = open(log_file_path, "a")
    mod_str_file = open(module_strings_path, "a")

    # test the networks
    files_list = []
    for net_file in os.listdir(args.subnets_folder):
        if net_file in ["test_log.txt", "module_strings.txt"]:
            pass
        else:
            files_list.append(net_file)

    sorted_files_list = sorted(files_list, key=lambda k: int(k[7:-5]))

    for net_file in sorted_files_list:
        # load the configuration of the network to evaluate from input arguments
        net_file_path = os.path.join(args.subnets_folder, net_file)
        subnet_file = open(net_file_path, "r")
        subnet_dict = json.load(subnet_file)
        arch = subnet_dict["arch"]
        img_size = arch.pop("r")

        # extract the subnet with the structure defined in the json from the supernet
        supernet.set_active_subnet(**arch)
        subnet = supernet.get_active_subnet(preserve_weight=True)

        # select test_loader
        data_provider.assign_active_img_size(img_size)
        test_loader = data_provider.test

        sdl = data_provider.build_sub_train_loader(data_provider.subset_size, data_provider.subset_batch_size)
        set_running_statistics(subnet, sdl)

        # test the model
        subnet = subnet.cuda()
        criterion = torch.nn.CrossEntropyLoss()
        loss, top1, top5 = validate(subnet, test_loader, criterion, net_file)
        subnet_file.close()

        _, net_file_name = os.path.split(net_file)

        subnet_dict["arch"]["r"] = img_size
        test_results = {
            "loss": round(loss, 4),
            "top1": round(top1, 4),
            "err1": round(100 - top1, 4),
            "top5": round(top5, 4),
            "err5": round(100 - top5, 4)
        }
        subnet_dict["test"] = test_results

        # log results
        log_str = "Network in " + net_file_name + ":\n"
        log_str += json.dumps(subnet_dict) + "\n\n"
        log_file.write(log_str)

        # log networks structures
        mod_str = "Network in " + net_file_name + ":\n"
        mod_str += subnet.module_str + "\n----------------------------\n\n"
        mod_str_file.write(mod_str)

    log_file.close()
    mod_str_file.close()
    return


# parse command line input
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # supernet related settings
    parser.add_argument(
        "--supernet",
        type=str,
        default="OFAMobileNetV3",
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
        '--search_type',
        type=str,
        default="ea",
        choices=["ea"],
        help='search method to be used',
    )

    # data related settings
    parser.add_argument(
        '--dataset',
        type=str,
        default='tiny_imagenet',
        help='name of the dataset',
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
            "stl10",
        ]
    )
    parser.add_argument('--dataset_path', type=str, default=None, help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=100, help='test batch size for inference')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for data loading')

    # model related settings
    parser.add_argument('--weights_path', type=str, help="supernet pretrained weights (checkpoint.pth.tar) file path")
    parser.add_argument('--subnets_folder', type=str, help="directory containing subnets")

    args = parser.parse_args()

    assign_dataset_arguments(args)

    main(args)