import torch
from ofa.utils import get_net_device
from tqdm import tqdm

from .utils import *


def validate_se(net, data_loader, criterion, epoch=0, run_str='', no_logs=False, train_mode=False):

    if train_mode:
        net.train()
    else:
        net.eval()

    device = get_net_device(net)

    losses = AverageMeter()
    metric_dict = get_metric_dict()

    with torch.no_grad():
        with tqdm(
            total=len(data_loader),
            desc='Validate Epoch #{} {}'.format(epoch + 1, run_str),
            disable=no_logs,
        ) as t:

            for i, (images, labels) in enumerate(data_loader):
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


def validate_ee(net, data_loader, criterion, branches_weights, ensemble_weights, epoch=0, run_str='', no_logs=False, train_mode=False):

    if train_mode:
        net.train()
    else:
        net.eval()

    device = get_net_device(net)

    net_losses = AverageMeter()
    net_metric_dict = get_metric_dict()

    branches_losses = []
    branches_metric_dict = []
    for _ in range(len(branches_weights)):
        branches_losses.append(AverageMeter())
        branches_metric_dict.append(get_metric_dict())

    with torch.no_grad():
        with tqdm(
            total=len(data_loader),
            desc='Validate Epoch #{} {}'.format(epoch + 1, run_str),
            disable=no_logs,
        ) as t:

            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(device), labels.to(device)

                # compute output
                outputs = net(images)

                # compute weighted losses for the branches
                weighted_losses = []
                for output, weight in zip(outputs, branches_weights):
                    br_loss = criterion(output, labels) * weight
                    weighted_losses.append(br_loss)

                # compute and record network loss
                net_loss = 0
                for wl in weighted_losses:
                    net_loss += wl

                net_losses.update(net_loss.item(), images.size(0))

                # compute and record branches losses
                for weighted_loss, branch_loss in zip(weighted_losses, branches_losses):
                    branch_loss.update(weighted_loss.item(), images.size(0))

                # measure and record accuracy for branches
                for output, br_metric_dict in zip(outputs, branches_metric_dict):
                    update_metric(br_metric_dict, output, labels)

                # measure and record accuracy for network, weighted sum of outputs
                weighted_outputs = []
                for output, weight in zip(outputs, ensemble_weights):
                    weighted_outputs.append(output * weight)

                net_output = torch.stack(weighted_outputs)
                net_output = torch.sum(net_output, dim=0)
                update_metric(net_metric_dict, net_output, labels)

                # tqdm show
                t.set_postfix({
                    'net_loss': net_losses.avg,
                    **get_metric_vals(net_metric_dict, return_dict=True),
                    'img_size': images.size(2),
                })
                t.update(1)

    # compute returned values for network
    net_losses_avg = net_losses.avg
    net_metric_list_vals = get_metric_vals(net_metric_dict, return_dict=False)

    # compute returned values for branches
    branches_losses_avgs = [branch_losses.avg for branch_losses in branches_losses]
    branches_metric_list_vals = [get_metric_vals(br_metric_dict) for br_metric_dict in branches_metric_dict]
    br_top1s = [tops[0] for tops in branches_metric_list_vals]
    br_top5s = [tops[1] for tops in branches_metric_list_vals]
    branches_metric_list_vals = [br_top1s, br_top5s]

    return net_losses_avg, net_metric_list_vals, branches_losses_avgs, branches_metric_list_vals

