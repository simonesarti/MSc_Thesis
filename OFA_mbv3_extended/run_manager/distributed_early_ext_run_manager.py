import random
import time

import torch
import torch.nn.functional as F
from ofa.utils import DistributedMetric, AverageMeter, mix_images, mix_labels, cross_entropy_loss_with_soft_target, \
    list_mean
from tqdm import tqdm

from .new_distributed_run_manager import NewDistributedRunManager

__all__ = ["DistributedEarlyExitRunManager"]

from ..my_data_loaders.my_random_resize_crop import MyModRandomResizedCrop
from ..utils import reshape_list_axis0


class DistributedEarlyExitRunManager(NewDistributedRunManager):

    def __init__(
            self,
            path,
            net,
            run_config,
            hvd_compression,
            backward_steps=1,
            is_root=False,
            init=True
    ):

        super(DistributedEarlyExitRunManager, self).__init__(
            path,
            net,
            run_config,
            hvd_compression,
            backward_steps,
            is_root,
            init
        )

    """ train and test """
    # inherited "validate" used for validation in both training of full-net and progressive shrinking

    # used for the validation when training the full network
    def validate_full(
            self,
            epoch=0,
            is_test=False,
            run_str="",
            net=None,
            data_loader=None,
            no_logs=False,
    ):
        if net is None:
            net = self.net
        if data_loader is None:
            if is_test:
                data_loader = self.run_config.test_loader
            else:
                data_loader = self.run_config.valid_loader

        net.eval()

        # record values for whole network
        net_loss_meter = DistributedMetric("val_loss")
        net_metric_dict = self.get_metric_dict()

        # record values for singles branches
        branches_loss_meter = []
        branches_metric_dict = []
        for i in range(5):
            branches_loss_meter.append(DistributedMetric(f"branch{i+1}_val_loss"))
            branches_metric_dict.append(self.get_metric_dict())

        branches_weights = self.run_config.branches_weights
        ensemble_weights = self.run_config.ensemble_weights

        with torch.no_grad():
            with tqdm(
                    total=len(data_loader),
                    desc="Validate Epoch #{} {}".format(epoch + 1, run_str),
                    disable=no_logs or not self.is_root,
            ) as t:

                for i, (images, labels) in enumerate(data_loader):

                    images, labels = images.cuda(), labels.cuda()

                    # compute outputs, list of output tensors for the exits
                    outputs = net(images)

                    # compute loss for each branch
                    branches_losses = [self.test_criterion(output, labels) for output in outputs]

                    # compute weighted loss for each branch
                    weighted_losses = []
                    for branch_loss, branch_weight in zip(branches_losses, branches_weights):
                        weighted_losses.append(branch_loss * branch_weight)

                    # add the weighted losses together to get the network loss
                    net_loss = 0
                    for weighted_loss in weighted_losses:
                        net_loss += weighted_loss

                    # compute weighted output for each branch
                    weighted_outputs = []
                    for output, weight in zip(outputs, ensemble_weights):
                        weighted_outputs.append(output * weight)

                    # use the weighted outputs and sum to obtain the net output
                    net_output = torch.stack(weighted_outputs)
                    net_output = torch.sum(net_output, dim=0)

                    # measure net accuracy
                    self.update_metric(net_metric_dict, net_output, labels)

                    # measure branches accuracies
                    for branch_metric_dict, output in zip(branches_metric_dict, outputs):
                        self.update_metric(branch_metric_dict, output, labels)

                    # record net loss
                    net_loss_meter.update(net_loss, images.size(0))

                    # record branches losses
                    for branch_loss_meter, weighted_loss in zip(branches_loss_meter, weighted_losses):
                        branch_loss_meter.update(weighted_loss, images.size(0))

                    # set postfix
                    t.set_postfix(
                        {
                            "net_loss": net_loss_meter.avg.item(),
                            **self.get_metric_vals(net_metric_dict, return_dict=True),
                            "img_size": images.size(2),
                        }
                    )
                    t.update(1)

        # =======================
        # compute values to return

        # compute average loss for the network
        net_loss_avg = net_loss_meter.avg.item()

        # compute average losses for the branches
        branches_loss_avg = [branch_loss_meter.avg.item() for branch_loss_meter in branches_loss_meter]

        # get accuracy metrics for the network
        net_metric_vals_list = self.get_metric_vals(net_metric_dict)

        # get accuracy metrics for the branches
        branches_metric_vals_list_couples = [self.get_metric_vals(branch_metric_dict)
                                             for branch_metric_dict in branches_metric_dict]

        top1s = [top[0] for top in branches_metric_vals_list_couples]
        top5s = [top[1] for top in branches_metric_vals_list_couples]
        branches_metric_vals_list = [top1s, top5s]

        return net_loss_avg, branches_loss_avg, net_metric_vals_list, branches_metric_vals_list

    def validate_all_resolution(self, epoch=0, is_test=False, net=None):

        if net is None:
            net = self.net

        if isinstance(self.run_config.data_provider.image_size, list):

            img_size_list = []

            net_loss_list = []
            net_top1_list = []
            net_top5_list = []

            branches_losses_list = []
            branches_top1s_list = []
            branches_top5s_list = []

            for img_size in self.run_config.data_provider.image_size:
                # set image size
                img_size_list.append(img_size)
                self.run_config.data_provider.assign_active_img_size(img_size)
                self.reset_running_statistics(
                    net=net,
                    subset_size=self.run_config.data_provider.subset_size,
                    subset_batch_size=self.run_config.data_provider.subset_batch_size
                )
                run_str = f"for image size {img_size}"
                # compute losses and accuracies
                net_loss, branches_losses, net_tops, branches_tops = self.validate_full(
                    epoch=epoch,
                    is_test=is_test,
                    run_str=run_str,
                    net=net
                )
                net_loss_list.append(net_loss)
                net_top1_list.append(net_tops[0])
                net_top5_list.append(net_tops[1])
                branches_losses_list.append(branches_losses)
                branches_top1s_list.append(branches_tops[0])
                branches_top5s_list.append(branches_tops[1])

            return (
                img_size_list,
                net_loss_list,
                net_top1_list,
                net_top5_list,
                branches_losses_list,
                branches_top1s_list,
                branches_top5s_list
            )

        else:
            run_str = f"for image size {self.run_config.data_provider.image_size}"
            net_loss, branches_losses, net_tops, branches_tops = self.validate_full(
                epoch=epoch,
                is_test=is_test,
                run_str=run_str,
                net=net
            )

            return (
                [self.run_config.data_provider.active_img_size],
                [net_loss],
                [net_tops[0]],
                [net_tops[1]],
                [branches_losses],
                [branches_tops[0]],
                [branches_tops[1]],
            )

    def train_one_epoch(self, args, epoch, warmup_epochs=0, warmup_lr=0):

        # switch to train mode
        self.net.train()
        self.run_config.train_loader.sampler.set_epoch(epoch)  # required by distributed sampler
        MyModRandomResizedCrop.EPOCH = epoch  # required by elastic resolution

        n_batch = len(self.run_config.train_loader)

        net_loss_meter = DistributedMetric("train_loss")
        net_metric_dict = self.get_metric_dict()

        # record values for singles branches
        branches_loss_meter = []
        branches_metric_dict = []
        for i in range(5):
            branches_loss_meter.append(DistributedMetric(f"branch{i + 1}_train_loss"))
            branches_metric_dict.append(self.get_metric_dict())

        branches_weights = self.run_config.branches_weights
        ensemble_weights = self.run_config.ensemble_weights

        data_time = AverageMeter()

        with tqdm(
                total=n_batch,
                desc="{} Train Epoch #{}".format(self.run_config.dataset, epoch + 1),
                disable=not self.is_root,
        ) as t:

            end = time.time()
            for i, (images, labels) in enumerate(self.run_config.train_loader):
                MyModRandomResizedCrop.BATCH = i
                data_time.update(time.time() - end)
                if epoch < warmup_epochs:
                    new_lr = self.run_config.warmup_adjust_learning_rate(
                        optimizer=self.optimizer,
                        T_total=warmup_epochs * n_batch,
                        nBatch=n_batch,
                        epoch=epoch,
                        batch=i,
                        warmup_lr=warmup_lr,
                    )
                else:
                    new_lr = self.run_config.adjust_learning_rate(
                        optimizer=self.optimizer,
                        epoch=epoch - warmup_epochs,
                        batch=i,
                        nBatch=n_batch
                    )

                images, labels = images.cuda(), labels.cuda()
                target = labels

                if isinstance(self.run_config.mixup_alpha, float):
                    # transform data
                    lam = random.betavariate(self.run_config.mixup_alpha, self.run_config.mixup_alpha)
                    images = mix_images(images, lam)
                    labels = mix_labels(labels, lam, self.run_config.data_provider.n_classes,
                                        self.run_config.label_smoothing)

                # soft target
                if args.teacher_model is not None:  # if not None, always Early Exit type
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

                # compute outputs
                outputs = self.net(images)

                # compute loss for each branch
                branches_losses = [self.train_criterion(output, labels) for output in outputs]

                # compute weighted loss for each branch
                weighted_losses = []
                for branch_loss, branch_weight in zip(branches_losses, branches_weights):
                    weighted_losses.append(branch_loss * branch_weight)

                # add the losses together
                net_loss = 0
                for weighted_loss in weighted_losses:
                    net_loss += weighted_loss

                if args.teacher_model is None:
                    loss_type = "ce"
                else:
                    if args.kd_type == "ce":

                        weighted_kd_losses = [cross_entropy_loss_with_soft_target(output, teacher_soft_label) * w
                                              for output, w in zip(outputs, branches_weights)]
                    else:
                        weighted_kd_losses = [F.mse_loss(output, teacher_soft_logits) * w
                                              for output, w in zip(outputs, branches_weights)]

                    kd_loss = 0
                    for idx, w_kd_l in enumerate(weighted_kd_losses):
                        kd_loss += w_kd_l  # count the total kd_loss
                        weighted_losses[idx] += w_kd_l * args.kd_ratio  # add in branches losses

                    net_loss += kd_loss * args.kd_ratio  # add in total loss
                    loss_type = "%.1fkd+ce" % args.kd_ratio

                # compute gradient and do optimizer step
                self.optimizer.zero_grad()  # or self.net.zero_grad()
                net_loss.backward()
                self.optimizer.step()

                # compute weighted output for each branch
                weighted_outputs = []
                for output, weight in zip(outputs, ensemble_weights):
                    weighted_outputs.append(output * weight)

                # use the weighted outputs and sum to obtain the net output
                net_output = torch.stack(weighted_outputs)
                net_output = torch.sum(net_output, dim=0)

                # measure net accuracy
                self.update_metric(net_metric_dict, net_output, target)

                # measure branches accuracies
                for branch_metric_dict, output in zip(branches_metric_dict, outputs):
                    self.update_metric(branch_metric_dict, output, target)

                # record net loss
                net_loss_meter.update(net_loss, images.size(0))

                # record branches losses
                for branch_loss_meter, weighted_loss in zip(branches_loss_meter, weighted_losses):
                    branch_loss_meter.update(weighted_loss, images.size(0))

                # set postfix
                t.set_postfix(
                    {
                        "img_size": images.size(2),
                        "lr": new_lr,
                        "loss_type": loss_type,
                        "net_loss": net_loss_meter.avg.item(),
                        **self.get_metric_vals(net_metric_dict, return_dict=True),
                        "data_time": data_time.avg,
                    }
                )
                t.update(1)
                end = time.time()

        # =======================
        # compute values to return

        # compute average loss for the network
        net_loss_avg = net_loss_meter.avg.item()

        # compute average losses for the branches
        branches_loss_avg = [branch_loss_meter.avg.item() for branch_loss_meter in branches_loss_meter]

        # get accuracy metrics for the network
        net_metric_vals_list = self.get_metric_vals(net_metric_dict)

        # get accuracy metrics for the branches
        branches_metric_vals_list_couples = [self.get_metric_vals(branch_metric_dict)
                                             for branch_metric_dict in branches_metric_dict]

        top1s = [top[0] for top in branches_metric_vals_list_couples]
        top5s = [top[1] for top in branches_metric_vals_list_couples]
        branches_metric_vals_list = [top1s, top5s]

        return net_loss_avg, branches_loss_avg, net_metric_vals_list, branches_metric_vals_list

    def train(self, args, warmup_epochs=5, warmup_lr=0):
        # early_stopping_meter = AccEarlyStoppingMeter(patience=args.patience) #esm
        # should_stop = False #esm
        for epoch in range(self.start_epoch, self.run_config.n_epochs + warmup_epochs):
            net_train_loss, branches_train_loss, net_train_tops, branches_train_tops = self.train_one_epoch(
                args=args,
                epoch=epoch,
                warmup_epochs=warmup_epochs,
                warmup_lr=warmup_lr
            )

            (img_size_list,
             net_val_loss_list,
             net_val_top1_list,
             net_val_top5_list,
             branches_val_losses_list,
             branches_val_top1s_list,
             branches_val_top5s_list) = self.validate_all_resolution(epoch=epoch, is_test=False)

            is_best = list_mean(net_val_top1_list) > self.best_acc
            self.best_acc = max(self.best_acc, list_mean(net_val_top1_list))

            if self.is_root:

                # write logs
                val_log = f"VALID EPOCH [{epoch + 1 - warmup_epochs}/{self.run_config.n_epochs}]\n\n"

                val_log += f"Validation:\n"
                val_log += f"Net loss: {list_mean(net_val_loss_list):.3f}\n"
                val_log += f"Net top1: {list_mean(net_val_top1_list):.3f} (Best acc: {self.best_acc:.3f})\n"
                val_log += f"Net top5: {list_mean(net_val_top5_list):.3f}\n"

                val_log += "Resolutions Net top1s: "
                for i_s, v_a in zip(img_size_list, net_val_top1_list):
                    val_log += "(%d, %.3f), " % (i_s, v_a)
                val_log += "\n"

                val_log += f"Branches losses: ["
                branches_val_losses_list = reshape_list_axis0(branches_val_losses_list)
                losses_mean = [list_mean(elem) for elem in branches_val_losses_list]
                for lm in losses_mean:
                    val_log += f"{lm:.3f}, "
                val_log += "]\n"

                val_log += f"Branches top1: ["
                branches_val_top1s_list = reshape_list_axis0(branches_val_top1s_list)
                top1_mean = [list_mean(elem) for elem in branches_val_top1s_list]
                for t1m in top1_mean:
                    val_log += f"{t1m:.3f}, "
                val_log += "]\n"

                val_log += f"Branches top5: ["
                branches_val_top5s_list = reshape_list_axis0(branches_val_top5s_list)
                top5_mean = [list_mean(elem) for elem in branches_val_top5s_list]
                for t5m in top5_mean:
                    val_log += f"{t5m:.3f}, "
                val_log += "]\n\n"

                val_log += f"Training:\n"
                val_log += f"Net loss: {net_train_loss:.3f}\n"
                val_log += f"Net top1: {net_train_tops[0]:.3f}\n"
                val_log += f"Net top5: {net_train_tops[1]:.3f}\n"
                val_log += f"Branches losses: ["

                for btl in branches_train_loss:
                    val_log += f"{btl:.3f}, "
                val_log += "]\n"
                val_log += f"Branches top1: ["
                for btt1 in branches_train_tops[0]:
                    val_log += f"{btt1:.3f}, "
                val_log += "]\n"
                val_log += f"Branches top5: ["
                for btt5 in branches_train_tops[1]:
                    val_log += f"{btt5:.3f}, "
                val_log += "]\n"

                val_log += "\n\n\n"

                self.write_log(val_log, prefix="valid", should_print=False)

                # should_stop = early_stopping_meter.update(list_mean(net_val_top1_list)) #esm

                self.save_model(
                    {
                        "epoch": epoch,
                        "best_acc": self.best_acc,
                        "optimizer": self.optimizer.state_dict(),
                        "state_dict": self.net.state_dict(),
                    },
                    is_best=is_best,
                )

                # if should_stop: #esm
                #    return #esm

