import json
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from ofa.utils import (
    DistributedMetric,
    accuracy,
    write_log,
    init_models,
)
from ofa.utils import cross_entropy_loss_with_soft_target, cross_entropy_with_label_smoothing
from tqdm import tqdm

from OFA_mbv3_extended.utils.pytorch_utils import get_net_info

__all__ = ["NewDistributedRunManager"]


class NewDistributedRunManager:

    def __init__(
        self,
        path,
        net,
        run_config,
        hvd_compression,
        backward_steps=1,
        is_root=False,
        init=True,
    ):
        import horovod.torch as hvd

        self.path = path
        self.net = net
        self.run_config = run_config

        self.best_acc = 0.0
        self.start_epoch = 0

        self.is_root = is_root

        # create experiment path
        os.makedirs(self.path, exist_ok=True)

        # move network to gpu
        self.net.cuda()
        cudnn.benchmark = True

        # initialize network weights if root
        if init and self.is_root:
            init_models(self.net, self.run_config.model_init)

        # print experiment info if root
        # NOTE: skip connection sums are not counted in total_ops,
        # multiple ways but always imprecise to compute the FLOPs
        # params take into account all exits of the model in dyn nets, also the ones not currently active (ok)
        net_info = get_net_info(
            net=self.net,
            input_shape=self.run_config.data_provider.data_shape,
            print_info=True
        )
        with open("%s/net_info.txt" % self.path, "w") as fout:
            fout.write(json.dumps(net_info, indent=4) + "\n")
            try:
                fout.write(self.net.module_str + "\n")
            except Exception:
                fout.write("%s do not support `module_str`" % type(self.net))
            fout.write("%s\n" % self.run_config.data_provider.train.dataset.transform)
            fout.write("%s\n" % self.run_config.data_provider.test.dataset.transform)
            fout.write("%s\n" % self.net)

        # criterion
        if isinstance(self.run_config.mixup_alpha, float):
            self.train_criterion = cross_entropy_loss_with_soft_target
        elif self.run_config.label_smoothing > 0:
            self.train_criterion = (
                lambda pred, target: cross_entropy_with_label_smoothing(
                    pred, target, self.run_config.label_smoothing
                )
            )
        else:
            self.train_criterion = nn.CrossEntropyLoss()
        self.test_criterion = nn.CrossEntropyLoss()

        # optimizer
        if self.run_config.no_decay_keys:
            keys = self.run_config.no_decay_keys.split("#")
            net_params = [
                self.net.get_parameters(
                    keys, mode="exclude"
                ),  # parameters with weight decay
                self.net.get_parameters(
                    keys, mode="include"
                ),  # parameters without weight decay
            ]
        else:
            # noinspection PyBroadException
            try:
                net_params = self.network.weight_parameters()
            except Exception:
                net_params = []
                for param in self.network.parameters():
                    if param.requires_grad:
                        net_params.append(param)
        self.optimizer = self.run_config.build_optimizer(net_params)
        self.optimizer = hvd.DistributedOptimizer(
            self.optimizer,
            named_parameters=self.net.named_parameters(),
            compression=hvd_compression,
            backward_passes_per_step=backward_steps,
        )

    """ save path and log path """

    @property
    def save_path(self):
        if self.__dict__.get("_save_path", None) is None:
            save_path = os.path.join(self.path, "checkpoint")
            os.makedirs(save_path, exist_ok=True)
            self.__dict__["_save_path"] = save_path
        return self.__dict__["_save_path"]

    @property
    def logs_path(self):
        if self.__dict__.get("_logs_path", None) is None:
            logs_path = os.path.join(self.path, "logs")
            os.makedirs(logs_path, exist_ok=True)
            self.__dict__["_logs_path"] = logs_path
        return self.__dict__["_logs_path"]

    @property
    def network(self):
        return self.net

    @network.setter
    def network(self, new_val):
        self.net = new_val

    def write_log(self, log_str, prefix="valid", should_print=True, mode="a"):
        if self.is_root:
            write_log(self.logs_path, log_str, prefix, should_print, mode)

    """ save & load model & save_config & broadcast """

    def save_config(self, extra_run_config=None, extra_net_config=None):
        if self.is_root:
            run_save_path = os.path.join(self.path, "run.config")
            if not os.path.isfile(run_save_path):
                run_config = self.run_config.config
                if extra_run_config is not None:
                    run_config.update(extra_run_config)
                json.dump(run_config, open(run_save_path, "w"), indent=4)
                print("Run configs dump to %s" % run_save_path)

            try:
                net_save_path = os.path.join(self.path, "net.config")
                net_config = self.net.config
                if extra_net_config is not None:
                    net_config.update(extra_net_config)
                json.dump(net_config, open(net_save_path, "w"), indent=4)
                print("Network configs dump to %s" % net_save_path)
            except Exception:
                print("%s do not support net config" % type(self.net))

    def save_model(self, checkpoint=None, is_best=False, model_name=None):
        if self.is_root:
            if checkpoint is None:
                checkpoint = {"state_dict": self.net.state_dict()}

            if model_name is None:
                model_name = "checkpoint.pth.tar"

            latest_fname = os.path.join(self.save_path, "latest.txt")
            model_path = os.path.join(self.save_path, model_name)
            with open(latest_fname, "w") as _fout:
                _fout.write(model_path + "\n")
            torch.save(checkpoint, model_path)

            if is_best:
                best_path = os.path.join(self.save_path, "model_best.pth.tar")
                torch.save({"state_dict": checkpoint["state_dict"]}, best_path)

    def load_model(self, model_fname=None):
        if self.is_root:
            latest_fname = os.path.join(self.save_path, "latest.txt")
            if model_fname is None and os.path.exists(latest_fname):
                with open(latest_fname, "r") as fin:
                    model_fname = fin.readline()
                    if model_fname[-1] == "\n":
                        model_fname = model_fname[:-1]
            # noinspection PyBroadException
            try:
                if model_fname is None or not os.path.exists(model_fname):
                    model_fname = "%s/checkpoint.pth.tar" % self.save_path
                    with open(latest_fname, "w") as fout:
                        fout.write(model_fname + "\n")
                print("=> loading checkpoint '{}'".format(model_fname))
                checkpoint = torch.load(model_fname, map_location="cpu")
            except Exception:
                self.write_log(
                    "fail to load checkpoint from %s" % self.save_path, "valid"
                )
                return

            self.net.load_state_dict(checkpoint["state_dict"])
            if "epoch" in checkpoint:
                self.start_epoch = checkpoint["epoch"] + 1
            if "best_acc" in checkpoint:
                self.best_acc = checkpoint["best_acc"]
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])

            self.write_log("=> loaded checkpoint '{}'".format(model_fname), "valid")

    # noinspection PyArgumentList
    def broadcast(self):
        import horovod.torch as hvd

        self.start_epoch = hvd.broadcast(
            torch.LongTensor(1).fill_(self.start_epoch)[0], 0, name="start_epoch"
        ).item()
        self.best_acc = hvd.broadcast(
            torch.Tensor(1).fill_(self.best_acc)[0], 0, name="best_acc"
        ).item()
        hvd.broadcast_parameters(self.net.state_dict(), 0)
        hvd.broadcast_optimizer_state(self.optimizer, 0)

    """ metric related """

    def get_metric_dict(self):
        return {
            "top1": DistributedMetric("top1"),
            "top5": DistributedMetric("top5"),
        }

    def update_metric(self, metric_dict, output, labels):
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        metric_dict["top1"].update(acc1[0], output.size(0))
        metric_dict["top5"].update(acc5[0], output.size(0))

    def get_metric_vals(self, metric_dict, return_dict=False):
        if return_dict:
            return {key: metric_dict[key].avg.item() for key in metric_dict}
        else:
            return [metric_dict[key].avg.item() for key in metric_dict]

    def get_metric_names(self):
        return "top1", "top5"

    def reset_running_statistics(
        self, net=None, subset_size=2000, subset_batch_size=200, data_loader=None
    ):
        from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics

        if net is None:
            net = self.net
        if data_loader is None:
            data_loader = self.run_config.random_sub_train_loader(
                subset_size, subset_batch_size
            )
        set_running_statistics(net, data_loader)

    """ train and test """

    def validate(
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

        losses = DistributedMetric("val_loss")
        metric_dict = self.get_metric_dict()

        with torch.no_grad():
            with tqdm(
                    total=len(data_loader),
                    desc="Validate Epoch #{} {}".format(epoch + 1, run_str),
                    disable=no_logs or not self.is_root,
            ) as t:
                for i, (images, labels) in enumerate(data_loader):
                    images, labels = images.cuda(), labels.cuda()
                    # compute output
                    output = net(images)
                    loss = self.test_criterion(output, labels)
                    # measure accuracy and record loss
                    losses.update(loss, images.size(0))
                    self.update_metric(metric_dict, output, labels)
                    t.set_postfix(
                        {
                            "loss": losses.avg.item(),
                            **self.get_metric_vals(metric_dict, return_dict=True),
                            "img_size": images.size(2),
                        }
                    )
                    t.update(1)
        return losses.avg.item(), self.get_metric_vals(metric_dict)

    def validate_all_resolution(self, epoch=0, is_test=False, net=None):
        raise NotImplementedError

    def train_one_epoch(self, args, epoch, warmup_epochs=0, warmup_lr=0):
        raise NotImplementedError

    def train(self, args, warmup_epoch=0, warmup_lr=0):
        raise NotImplementedError

