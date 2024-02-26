import random
import time

import torch
import torch.nn.functional as F
from ofa.utils import DistributedMetric, AverageMeter, mix_images, mix_labels, cross_entropy_loss_with_soft_target, \
    list_mean
from tqdm import tqdm

from .new_distributed_run_manager import NewDistributedRunManager
from ..my_data_loaders.my_random_resize_crop import MyModRandomResizedCrop

__all__ = ["DistributedSingleExitRunManager"]


class DistributedSingleExitRunManager(NewDistributedRunManager):

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

        super(DistributedSingleExitRunManager, self).__init__(
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

    def validate_all_resolution(self, epoch=0, is_test=False, net=None):
        if net is None:
            net = self.net
        if isinstance(self.run_config.data_provider.image_size, list):
            img_size_list, loss_list, top1_list, top5_list = [], [], [], []
            for img_size in self.run_config.data_provider.image_size:
                img_size_list.append(img_size)
                self.run_config.data_provider.assign_active_img_size(img_size)
                self.reset_running_statistics(
                    net=net,
                    subset_size=self.run_config.data_provider.subset_size,
                    subset_batch_size=self.run_config.data_provider.subset_batch_size
                )
                run_str = f"for image size {img_size}"
                loss, (top1, top5) = self.validate(epoch=epoch, is_test=is_test, run_str=run_str, net=net)
                loss_list.append(loss)
                top1_list.append(top1)
                top5_list.append(top5)
            return img_size_list, loss_list, top1_list, top5_list
        else:
            loss, (top1, top5) = self.validate(epoch, is_test, net=net)
            return (
                [self.run_config.data_provider.active_img_size],
                [loss],
                [top1],
                [top5],
            )

    def train_one_epoch(self, args, epoch, warmup_epochs=5, warmup_lr=0):
        self.net.train()
        self.run_config.train_loader.sampler.set_epoch(epoch)   # required by distributed sampler
        MyModRandomResizedCrop.EPOCH = epoch  # required by elastic resolution

        n_batch = len(self.run_config.train_loader)

        losses = DistributedMetric("train_loss")
        metric_dict = self.get_metric_dict()
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
                        self.optimizer,
                        warmup_epochs * n_batch,
                        n_batch,
                        epoch,
                        i,
                        warmup_lr,
                    )
                else:
                    new_lr = self.run_config.adjust_learning_rate(self.optimizer, epoch - warmup_epochs, i, n_batch)

                images, labels = images.cuda(), labels.cuda()
                target = labels
                if isinstance(self.run_config.mixup_alpha, float):
                    # transform data
                    random.seed(int("%d%.3d" % (i, epoch)))
                    lam = random.betavariate(self.run_config.mixup_alpha, self.run_config.mixup_alpha)
                    images = mix_images(images, lam)
                    labels = mix_labels(labels, lam, self.run_config.data_provider.n_classes, self.run_config.label_smoothing)

                # soft target
                if args.teacher_model is not None:
                    args.teacher_model.train()
                    with torch.no_grad():
                        soft_logits = args.teacher_model(images).detach()
                        soft_label = F.softmax(soft_logits, dim=1)

                # compute output
                output = self.net(images)
                loss = self.train_criterion(output, labels)

                if args.teacher_model is None:
                    loss_type = "ce"
                else:
                    if args.kd_type == "ce":
                        kd_loss = cross_entropy_loss_with_soft_target(output, soft_label)
                    else:
                        kd_loss = F.mse_loss(output, soft_logits)
                    loss = args.kd_ratio * kd_loss + loss
                    loss_type = "%.1fkd+ce" % args.kd_ratio

                # update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure accuracy and record loss
                losses.update(loss, images.size(0))
                self.update_metric(metric_dict, output, target)
                t.set_postfix(
                    {
                        "loss": losses.avg.item(),
                        **self.get_metric_vals(metric_dict, return_dict=True),
                        "img_size": images.size(2),
                        "lr": new_lr,
                        "loss_type": loss_type,
                        "data_time": data_time.avg,
                    }
                )
                t.update(1)
                end = time.time()

            return losses.avg.item(), self.get_metric_vals(metric_dict)

    def train(self, args, warmup_epochs=5, warmup_lr=0):
        # early_stopping_meter = AccEarlyStoppingMeter(patience=args.patience) #esm
        # should_stop = False #esm

        for epoch in range(self.start_epoch, self.run_config.n_epochs + warmup_epochs):
            train_loss, (train_top1, train_top5) = self.train_one_epoch(args, epoch, warmup_epochs, warmup_lr)

            img_size, val_loss, val_acc, val_acc5 = self.validate_all_resolution(epoch=epoch, is_test=False)

            is_best = list_mean(val_acc) > self.best_acc
            self.best_acc = max(self.best_acc, list_mean(val_acc))
            if self.is_root:
                val_log = (
                    "[{0}/{1}]\tloss {2:.3f}\t{6} acc {3:.3f} ({4:.3f})\t{7} acc {5:.3f}\t"
                    "Train {6} {top1:.3f}\tloss {train_loss:.3f}\t".format(
                        epoch + 1 - warmup_epochs,
                        self.run_config.n_epochs,
                        list_mean(val_loss),
                        list_mean(val_acc),
                        self.best_acc,
                        list_mean(val_acc5),
                        *self.get_metric_names(),
                        top1=train_top1,
                        train_loss=train_loss
                    )
                )
                for i_s, v_a in zip(img_size, val_acc):
                    val_log += "(%d, %.3f), " % (i_s, v_a)
                self.write_log(val_log, prefix="valid", should_print=False)

                # compare val_acc to previous one, should_stop=True if decremented for "patience" times
                # should_stop = early_stopping_meter.update(list_mean(val_acc)) #esm

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




