import random
import time

import numpy as np
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
from ofa.utils import AverageMeter, mix_images, mix_labels, cross_entropy_loss_with_soft_target
from tqdm import tqdm

from OFA_mbv3_extended.my_data_loaders.my_random_resize_crop import MyModRandomResizedCrop
from .new_run_manager import NewRunManager

__all__ = ["SingleExitRunManager"]


class SingleExitRunManager(NewRunManager):

    def __init__(
            self,
            path,
            net,
            run_config,
            init=True,
            measure_latency=None,
            no_gpu=False
    ):
        super(SingleExitRunManager, self).__init__(
            path,
            net,
            run_config,
            init,
            measure_latency,
            no_gpu
        )

    """ train and test """

    # inherited "validate" used for validation in both training of full-net and progressive shrinking

    def validate_all_resolution(self, epoch=0, is_test=False, net=None):
        if net is None:
            net = self.network
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

    def train_one_epoch(self, args, epoch, warmup_epochs=0, warmup_lr=0):
        # switch to train mode
        self.net.train()
        MyModRandomResizedCrop.EPOCH = epoch  # required by elastic resolution

        n_batch = len(self.run_config.train_loader)

        losses = AverageMeter()
        metric_dict = self.get_metric_dict()
        data_time = AverageMeter()

        with tqdm(
                total=n_batch,
                desc="{} Train Epoch #{}".format(self.run_config.dataset, epoch + 1),
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

                images, labels = images.to(self.device), labels.to(self.device)
                target = labels
                if isinstance(self.run_config.mixup_alpha, float):
                    # transform data
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

                # compute gradient and do SGD step
                self.net.zero_grad()  # or self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure accuracy and record loss
                losses.update(loss.item(), images.size(0))
                self.update_metric(metric_dict, output, target)

                t.set_postfix(
                    {
                        "loss": losses.avg,
                        **self.get_metric_vals(metric_dict, return_dict=True),
                        "img_size": images.size(2),
                        "lr": new_lr,
                        "loss_type": loss_type,
                        "data_time": data_time.avg,
                    }
                )
                t.update(1)
                end = time.time()
        return losses.avg, self.get_metric_vals(metric_dict)

    def train(self, args, warmup_epoch=0, warmup_lr=0):

        # early_stopping_meter = AccEarlyStoppingMeter(patience=args.patience) #esm
        # should_stop = False #esm

        for epoch in range(self.start_epoch, self.run_config.n_epochs + warmup_epoch):
            train_loss, (train_top1, train_top5) = self.train_one_epoch(args, epoch, warmup_epoch, warmup_lr)

            if (epoch + 1) % self.run_config.validation_frequency == 0:
                img_size, val_loss, val_acc, val_acc5 = self.validate_all_resolution(epoch=epoch, is_test=False)

                is_best = np.mean(val_acc) > self.best_acc
                self.best_acc = max(self.best_acc, np.mean(val_acc))
                val_log = "Valid [{0}/{1}]\tloss {2:.3f}\t{5} {3:.3f} ({4:.3f})".format(
                    epoch + 1 - warmup_epoch,
                    self.run_config.n_epochs,
                    np.mean(val_loss),
                    np.mean(val_acc),
                    self.best_acc,
                    self.get_metric_names()[0],
                )
                val_log += "\t{2} {0:.3f}\tTrain {1} {top1:.3f}\tloss {train_loss:.3f}\t".format(
                    np.mean(val_acc5),
                    *self.get_metric_names(),
                    top1=train_top1,
                    train_loss=train_loss
                )
                for i_s, v_a in zip(img_size, val_acc):
                    val_log += "(%d, %.3f), " % (i_s, v_a)
                self.write_log(val_log, prefix="valid", should_print=False)

                # compare val_acc to previous one, should_stop=True if decremented for "patience" times
                # should_stop = early_stopping_meter.update(np.mean(val_acc)) #esm

            else:
                is_best = False

            self.save_model(
                {
                    "epoch": epoch,
                    "best_acc": self.best_acc,
                    "optimizer": self.optimizer.state_dict(),
                    "state_dict": self.network.state_dict(),
                },
                is_best=is_best,
            )

            # if should_stop: #esm
            #    return #esm
