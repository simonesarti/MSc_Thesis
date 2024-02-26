import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from ofa.imagenet_classification.elastic_nn.training.progressive_shrinking import load_models
from ofa.utils import AverageMeter, cross_entropy_loss_with_soft_target
from ofa.utils import (
    DistributedMetric,
    list_mean,
    subset_mean,
    val2list,
)
from tqdm import tqdm
import numpy as np

from OFA_mbv3_extended.my_data_loaders.my_random_resize_crop import MyModRandomResizedCrop
from OFA_mbv3_extended.networks.nets.my_networks import get_valid_nw_keys

__all__ = [
    "validate",
    "train_one_epoch_se",
    "train",
    "train_one_epoch_se",
    "train_one_epoch_ee",
    "train_elastic_net_width",
    "train_elastic_net_depth",
    "train_elastic_depth",
    "train_elastic_expand",
]

from OFA_mbv3_extended.run_manager import DistributedEarlyExitRunManager, DistributedSingleExitRunManager


def validate(
    run_manager,
    epoch=0,
    is_test=False,
    image_size_list=None,
    ks_list=None,
    expand_ratio_list=None,
    depth_list=None,
    net_width_list=None,
    net_depth_list=None,
    width_mult_list=None,
    additional_setting=None,
):
    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    dynamic_net.eval()

    # if values are missing, take them from the network
    if image_size_list is None:
        image_size_list = val2list(run_manager.run_config.data_provider.image_size, 1)
    if ks_list is None:
        ks_list = dynamic_net.ks_list
    if expand_ratio_list is None:
        expand_ratio_list = dynamic_net.expand_ratio_list
    if depth_list is None:
        depth_list = dynamic_net.depth_list

    if width_mult_list is None:
        if "width_mult_list" in dynamic_net.__dict__:
            width_mult_list = list(range(len(dynamic_net.width_mult_list)))
        else:
            width_mult_list = [0]

    try:
        # original network, name is static method
        net_name = dynamic_net.name()
    except TypeError:
        # staged networks, name is attribute
        net_name = dynamic_net.name

    # if value is missing, take it from net only if is a parallel network
    if net_width_list is None and "P" in net_name:
        net_width_list = dynamic_net.net_width_list

    # if value is missing, take it from net only if is an early exit network
    if net_depth_list is None and "EE" in net_name:
        net_depth_list = dynamic_net.net_depth_list

    subnet_settings = []
    for d in depth_list:
        for e in expand_ratio_list:
            for k in ks_list:
                for w in width_mult_list:
                    for img_size in image_size_list:

                        if net_width_list is not None:
                            allowed_nw = get_valid_nw_keys(net_width_list)
                            for nw in allowed_nw:

                                if net_depth_list is not None:
                                    for nd in net_depth_list:
                                        subnet_settings.append([
                                            {
                                                "image_size": img_size,
                                                "d": d,
                                                "e": e,
                                                "ks": k,
                                                "nd": nd,
                                                "nw": nw,
                                                "w": w,
                                            },
                                            "R%s-K%s-NW%s-ND%s-D%s-E%s-W%s" % (img_size, k, nw, nd, d, e, w),
                                        ])
                                        ordering = lambda i: (i[0]["image_size"], i[0]["ks"], i[0]["nw"], i[0]["nd"], i[0]["d"], i[0]["e"], i[0]["w"])
                                else:
                                    subnet_settings.append([
                                        {
                                            "image_size": img_size,
                                            "d": d,
                                            "e": e,
                                            "ks": k,
                                            "nw": nw,
                                            "w": w,
                                        },
                                        "R%s-K%s-NW%s-D%s-E%s-W%s" % (img_size, k, nw, d, e, w),
                                    ])
                                    ordering = lambda i: (i[0]["image_size"], i[0]["ks"], i[0]["nw"], i[0]["d"], i[0]["e"], i[0]["w"])

                        else:
                            if net_depth_list is not None:
                                for nd in net_depth_list:
                                    subnet_settings.append([
                                        {
                                            "image_size": img_size,
                                            "d": d,
                                            "e": e,
                                            "ks": k,
                                            "nd": nd,
                                            "w": w,
                                        },
                                        "R%s-K%s-ND%s-D%s-E%s-W%s" % (img_size, k, nd, d, e, w),
                                    ])
                                    ordering = lambda i: (i[0]["image_size"], i[0]["ks"], i[0]["nd"], i[0]["d"], i[0]["e"], i[0]["w"])
                            else:
                                subnet_settings.append([
                                    {
                                        "image_size": img_size,
                                        "d": d,
                                        "e": e,
                                        "ks": k,
                                        "w": w,
                                    },
                                    "R%s-K%s-D%s-E%s-W%s" % (img_size, k, d, e, w),
                                ])
                                ordering = lambda i: (i[0]["image_size"], i[0]["ks"], i[0]["d"], i[0]["e"], i[0]["w"])

    if additional_setting is not None:
        subnet_settings += additional_setting

    n_subnets = len(subnet_settings)
    max_to_evaluate = run_manager.run_config.n_elastic_val

    # possibility to select a subset to validate if too many substrings present, saves time in validation
    if 0 < max_to_evaluate < n_subnets:
        subnet_settings = random.sample(subnet_settings, max_to_evaluate)

    subnet_settings = sorted(subnet_settings, key=ordering)

    losses_of_subnets, top1_of_subnets, top5_of_subnets = [], [], []

    valid_log = ""
    for setting, name in subnet_settings:
        run_manager.write_log("-" * 30 + " Validate %s " % name + "-" * 30, "train", should_print=False)
        run_manager.run_config.data_provider.assign_active_img_size(setting.pop("image_size"))
        dynamic_net.set_active_subnet(**setting)
        run_manager.write_log(dynamic_net.module_str, "train", should_print=False)

        run_manager.reset_running_statistics(
            net=dynamic_net,
            subset_size=run_manager.run_config.data_provider.subset_size,
            subset_batch_size=run_manager.run_config.data_provider.subset_batch_size
        )
        loss, (top1, top5) = run_manager.validate(epoch=epoch, is_test=is_test, run_str=name, net=dynamic_net)
        losses_of_subnets.append(loss)
        top1_of_subnets.append(top1)
        top5_of_subnets.append(top5)
        valid_log += "%s (%.3f), " % (name, top1)

    return (
        np.mean(losses_of_subnets),
        np.mean(top1_of_subnets),
        np.mean(top5_of_subnets),
        valid_log,
    )


def train_one_epoch_se(run_manager, args, epoch, warmup_epochs=0, warmup_lr=0):
    dynamic_net = run_manager.network
    distributed = isinstance(run_manager, DistributedSingleExitRunManager)

    # switch to train mode
    dynamic_net.train()
    if distributed:
        run_manager.run_config.train_loader.sampler.set_epoch(epoch)
    MyModRandomResizedCrop.EPOCH = epoch

    nBatch = len(run_manager.run_config.train_loader)

    data_time = AverageMeter()
    losses = DistributedMetric("train_loss") if distributed else AverageMeter()
    metric_dict = run_manager.get_metric_dict()

    with tqdm(
        total=nBatch,
        desc="Train Epoch #{}".format(epoch + 1),
        disable=distributed and not run_manager.is_root,
    ) as t:
        end = time.time()
        for i, (images, labels) in enumerate(run_manager.run_config.train_loader):
            MyModRandomResizedCrop.BATCH = i
            data_time.update(time.time() - end)
            if epoch < warmup_epochs:
                new_lr = run_manager.run_config.warmup_adjust_learning_rate(
                    run_manager.optimizer,
                    warmup_epochs * nBatch,
                    nBatch,
                    epoch,
                    i,
                    warmup_lr,
                )
            else:
                new_lr = run_manager.run_config.adjust_learning_rate(
                    run_manager.optimizer, epoch - warmup_epochs, i, nBatch
                )

            images, labels = images.cuda(), labels.cuda()
            target = labels

            # soft target
            if args.kd_ratio > 0:
                args.teacher_model.train()
                with torch.no_grad():
                    soft_logits = args.teacher_model(images).detach()
                    soft_label = F.softmax(soft_logits, dim=1)

            # clean gradients
            dynamic_net.zero_grad()

            loss_of_subnets = []
            # compute output
            subnet_str = ""
            for _ in range(args.dynamic_batch_size):
                # set random seed before sampling
                subnet_seed = int("%d%.3d%.3d" % (epoch * nBatch + i, _, 0))
                random.seed(subnet_seed)
                subnet_settings = dynamic_net.sample_active_subnet()
                subnet_str += (
                    "%d: " % _
                    + ",".join(
                        [
                            "%s_%s"
                            % (
                                key,
                                "%.1f" % subset_mean(val, 0)
                                if isinstance(val, list)
                                else val,
                            )
                            for key, val in subnet_settings.items()
                        ]
                    )
                    + " || "
                )

                output = run_manager.net(images)
                if args.kd_ratio == 0:
                    loss = run_manager.train_criterion(output, labels)
                    loss_type = "ce"
                else:
                    if args.kd_type == "ce":
                        kd_loss = cross_entropy_loss_with_soft_target(output, soft_label)
                    else:
                        kd_loss = F.mse_loss(output, soft_logits)

                    loss = args.kd_ratio * kd_loss + run_manager.train_criterion(output, labels)
                    loss_type = "%.1fkd-%s & ce" % (args.kd_ratio, args.kd_type)

                # measure accuracy and record loss
                loss_of_subnets.append(loss)
                run_manager.update_metric(metric_dict, output, target)

                # subnet backward, accumulate gradient
                loss.backward()

            run_manager.optimizer.step()

            losses.update(list_mean(loss_of_subnets), images.size(0))

            t.set_postfix(
                {
                    "loss": losses.avg.item(),
                    **run_manager.get_metric_vals(metric_dict, return_dict=True),
                    "R": images.size(2),
                    "lr": new_lr,
                    "loss_type": loss_type,
                    "seed": str(subnet_seed),
                    "str": subnet_str,
                    "data_time": data_time.avg,
                }
            )
            t.update(1)
            end = time.time()
    return losses.avg.item(), run_manager.get_metric_vals(metric_dict)


def train_one_epoch_ee(run_manager, args, epoch, warmup_epochs=0, warmup_lr=0):
    dynamic_net = run_manager.network

    distributed = isinstance(run_manager, DistributedEarlyExitRunManager)

    # switch to train mode
    dynamic_net.train()
    if distributed:
        run_manager.run_config.train_loader.sampler.set_epoch(epoch)
    MyModRandomResizedCrop.EPOCH = epoch

    nBatch = len(run_manager.run_config.train_loader)

    data_time = AverageMeter()
    losses = DistributedMetric("train_loss") if distributed else AverageMeter()
    metric_dict = run_manager.get_metric_dict()

    branches_weights = run_manager.run_config.branches_weights
    ensemble_weights = run_manager.run_config.ensemble_weights

    with tqdm(
        total=nBatch,
        desc="Train Epoch #{}".format(epoch + 1),
        disable=distributed and not run_manager.is_root,
    ) as t:
        end = time.time()
        for i, (images, labels) in enumerate(run_manager.run_config.train_loader):
            MyModRandomResizedCrop.BATCH = i
            data_time.update(time.time() - end)
            if epoch < warmup_epochs:
                new_lr = run_manager.run_config.warmup_adjust_learning_rate(
                    run_manager.optimizer,
                    warmup_epochs * nBatch,
                    nBatch,
                    epoch,
                    i,
                    warmup_lr,
                )
            else:
                new_lr = run_manager.run_config.adjust_learning_rate(
                    run_manager.optimizer,
                    epoch - warmup_epochs,
                    i,
                    nBatch
                )

            images, labels = images.cuda(), labels.cuda()
            target = labels

            # soft target
            if args.kd_ratio > 0:
                args.teacher_model.train()
                with torch.no_grad():
                    soft_logits_list = args.teacher_model(images)
                    for soft_logits in soft_logits_list:
                        soft_logits.detach()

                    # weight the logits
                    weighted_logits = []
                    for soft_logits, weight in zip(soft_logits_list, ensemble_weights):
                        weighted_logits.append(soft_logits * weight)

                    teacher_soft_logits = torch.stack(weighted_logits)
                    teacher_soft_logits = torch.sum(teacher_soft_logits, dim=0)

                    teacher_soft_label = F.softmax(teacher_soft_logits, dim=1)

            # clean gradients
            dynamic_net.zero_grad()

            loss_of_subnets = []
            # compute output
            subnet_str = ""
            for _ in range(args.dynamic_batch_size):
                # set random seed before sampling
                subnet_seed = int("%d%.3d%.3d" % (epoch * nBatch + i, _, 0))
                random.seed(subnet_seed)
                subnet_settings = dynamic_net.sample_active_subnet()
                subnet_str += (
                        "%d: " % _
                        + ",".join(
                    [
                        "%s_%s"
                        % (
                            key,
                            "%.1f" % subset_mean(val, 0)
                            if isinstance(val, list)
                            else val,
                        )
                        for key, val in subnet_settings.items()
                    ]
                )
                        + " || "
                )

                # sampled subnet has single output
                output = run_manager.net(images)

                # keep loss as is if no knowledge distillation
                if args.kd_ratio == 0:
                    loss = run_manager.train_criterion(output, labels)
                    loss_type = "ce"

                # teacher net has multiple exits, sum the loss of the ensemble of the branches
                else:
                    if args.kd_type == "ce":
                        kd_loss = cross_entropy_loss_with_soft_target(output, teacher_soft_label)
                    else:
                        kd_loss = F.mse_loss(output, teacher_soft_logits)

                    loss = args.kd_ratio * kd_loss + run_manager.train_criterion(output, labels)
                    loss_type = "%.1fkd-%s & ce" % (args.kd_ratio, args.kd_type)

                # measure accuracy and record loss
                loss_of_subnets.append(loss)
                run_manager.update_metric(metric_dict, output, target)

                # subnet backward, accumulate gradient
                loss.backward()

            run_manager.optimizer.step()

            losses.update(list_mean(loss_of_subnets), images.size(0))

            t.set_postfix(
                {
                    "loss": losses.avg.item(),
                    **run_manager.get_metric_vals(metric_dict, return_dict=True),
                    "R": images.size(2),
                    "lr": new_lr,
                    "loss_type": loss_type,
                    "seed": str(subnet_seed),
                    "str": subnet_str,
                    "data_time": data_time.avg,
                }
            )
            t.update(1)
            end = time.time()
    return losses.avg.item(), run_manager.get_metric_vals(metric_dict)


def train(run_manager, args, validate_func=None):

    # chose type of training depending on supernet type
    if "EE" in args.network:
        train_one_epoch = train_one_epoch_ee
        distributed = isinstance(run_manager, DistributedEarlyExitRunManager)
    else:
        train_one_epoch = train_one_epoch_se
        distributed = isinstance(run_manager, DistributedSingleExitRunManager)

    if validate_func is None:
        validate_func = validate

    # early_stopping_meter = AccEarlyStoppingMeter(patience=args.patience) #esm

    for epoch in range(run_manager.start_epoch, run_manager.run_config.n_epochs + args.warmup_epochs):
        train_loss, (train_top1, train_top5) = train_one_epoch(run_manager, args, epoch, args.warmup_epochs, args.warmup_lr)

        if (epoch + 1) % args.validation_frequency == 0:
            val_loss, val_acc, val_acc5, _val_log = validate_func(run_manager, epoch=epoch, is_test=False)
            # best_acc
            is_best = val_acc > run_manager.best_acc
            run_manager.best_acc = max(run_manager.best_acc, val_acc)

            if not distributed or run_manager.is_root:
                val_log = (
                    "Valid [{0}/{1}] loss={2:.3f}, top-1={3:.3f} ({4:.3f})".format(
                        epoch + 1 - args.warmup_epochs,
                        run_manager.run_config.n_epochs,
                        val_loss,
                        val_acc,
                        run_manager.best_acc,
                    )
                )
                val_log += ", Train top-1 {top1:.3f}, Train loss {loss:.3f}\t".format(top1=train_top1, loss=train_loss)
                val_log += _val_log
                run_manager.write_log(val_log, "valid", should_print=False)

                run_manager.save_model(
                    {
                        "epoch": epoch,
                        "best_acc": run_manager.best_acc,
                        "optimizer": run_manager.optimizer.state_dict(),
                        "state_dict": run_manager.network.state_dict(),
                    },
                    is_best=is_best,
                )

            # early stopping
            # should_stop = early_stopping_meter.update(val_acc) #esm
            # if should_stop: #esm
            #    return #esm


def train_elastic_net_width(train_func, run_manager, args, validate_func_dict):

    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    net_width_stage_list = dynamic_net.net_width_list.copy()
    net_width_stage_list.sort(reverse=True)
    n_stages = len(net_width_stage_list) - 1
    current_stage = n_stages - 1

    # load pretrained models
    if run_manager.start_epoch == 0 and not args.resume:
        validate_func_dict["net_width_list"] = sorted(dynamic_net.net_width_list)

        load_models(run_manager, dynamic_net, model_path=args.ofa_checkpoint_path)
        # validate after loading weights
        run_manager.write_log(
            "%.3f\t%.3f\t%.3f\t%s"
            % validate(run_manager, is_test=True, **validate_func_dict),
            "valid",
        )
    else:
        assert args.resume

    run_manager.write_log(
        "-" * 30
        + "Supporting Elastic Network Width: %s -> %s"
        % (net_width_stage_list[: current_stage + 1], net_width_stage_list[: current_stage + 2])
        + "-" * 30,
        "valid",
    )

    # during training, validate only on max and min net width values
    validate_func_dict["net_width_list"] = sorted({min(net_width_stage_list), max(net_width_stage_list)})

    # train
    train_func(
        run_manager,
        args,
        lambda _run_manager, epoch, is_test: validate(_run_manager, epoch, is_test, **validate_func_dict),
    )


def train_elastic_net_depth(train_func, run_manager, args, validate_func_dict):

    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    net_depth_stage_list = dynamic_net.net_depth_list.copy()
    net_depth_stage_list.sort(reverse=True)
    n_stages = len(net_depth_stage_list) - 1
    current_stage = n_stages - 1

    # load pretrained models
    if run_manager.start_epoch == 0 and not args.resume:
        validate_func_dict["net_depth_list"] = sorted(dynamic_net.net_depth_list)

        load_models(run_manager, dynamic_net, model_path=args.ofa_checkpoint_path)
        # validate after loading weights
        run_manager.write_log(
            "%.3f\t%.3f\t%.3f\t%s"
            % validate(run_manager, is_test=True, **validate_func_dict),
            "valid",
        )
    else:
        assert args.resume

    run_manager.write_log(
        "-" * 30
        + "Supporting Elastic Network Depth: %s -> %s"
        % (net_depth_stage_list[: current_stage + 1], net_depth_stage_list[: current_stage + 2])
        + "-" * 30,
        "valid",
    )

    # during training, validate only on max and min net depth values
    validate_func_dict["net_depth_list"] = sorted({min(net_depth_stage_list), max(net_depth_stage_list)})

    # train
    train_func(
        run_manager,
        args,
        lambda _run_manager, epoch, is_test: validate(_run_manager, epoch, is_test, **validate_func_dict),
    )


def train_elastic_depth(train_func, run_manager, args, validate_func_dict):

    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    depth_stage_list = dynamic_net.depth_list.copy()
    depth_stage_list.sort(reverse=True)
    n_stages = len(depth_stage_list) - 1
    current_stage = n_stages - 1

    # load pretrained models
    if run_manager.start_epoch == 0 and not args.resume:
        validate_func_dict["depth_list"] = sorted(dynamic_net.depth_list)

        load_models(run_manager, dynamic_net, model_path=args.ofa_checkpoint_path)
        # validate after loading weights
        run_manager.write_log(
            "%.3f\t%.3f\t%.3f\t%s"
            % validate(run_manager, is_test=True, **validate_func_dict),
            "valid",
        )
    else:
        assert args.resume

    run_manager.write_log(
        "-" * 30
        + "Supporting Elastic Depth: %s -> %s"
        % (depth_stage_list[: current_stage + 1], depth_stage_list[: current_stage + 2])
        + "-" * 30,
        "valid",
    )

    # during training, validate only on max and min depth values
    validate_func_dict["depth_list"] = sorted({min(depth_stage_list), max(depth_stage_list)})

    # train
    train_func(
        run_manager,
        args,
        lambda _run_manager, epoch, is_test: validate(_run_manager, epoch, is_test, **validate_func_dict),
    )


def train_elastic_expand(train_func, run_manager, args, validate_func_dict):

    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    expand_stage_list = dynamic_net.expand_ratio_list.copy()
    expand_stage_list.sort(reverse=True)
    n_stages = len(expand_stage_list) - 1
    current_stage = n_stages - 1

    # load pretrained models
    if run_manager.start_epoch == 0 and not args.resume:
        validate_func_dict["expand_ratio_list"] = sorted(dynamic_net.expand_ratio_list)

        load_models(run_manager, dynamic_net, model_path=args.ofa_checkpoint_path)
        dynamic_net.re_organize_middle_weights(expand_ratio_stage=current_stage)
        run_manager.write_log(
            "%.3f\t%.3f\t%.3f\t%s"
            % validate(run_manager, is_test=True, **validate_func_dict),
            "valid",
        )
    else:
        assert args.resume

    run_manager.write_log(
        "-" * 30
        + "Supporting Elastic Expand Ratio: %s -> %s"
        % (expand_stage_list[: current_stage + 1], expand_stage_list[: current_stage + 2])
        + "-" * 30,
        "valid",
    )

    validate_func_dict["expand_ratio_list"] = sorted({min(expand_stage_list), max(expand_stage_list)})

    # train
    train_func(
        run_manager,
        args,
        lambda _run_manager, epoch, is_test: validate(_run_manager, epoch, is_test, **validate_func_dict),
    )
