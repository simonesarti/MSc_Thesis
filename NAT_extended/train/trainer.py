# import sys
# sys.path.insert(0, './')

import itertools
import os
import random
from abc import ABC
# figure out if the model can be trained with mixed-precision
from contextlib import suppress

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics
from ofa.utils import cross_entropy_loss_with_soft_target, subset_mean, list_mean, get_net_device, AverageMeter
from timm.utils import ApexScaler, NativeScaler
from tqdm import tqdm

from NAT_extended.evaluation.evaluate import validate_se, validate_ee
from NAT_extended.evaluation.utils import get_metric_dict, update_metric, get_metric_vals
from NAT_extended.train.utils import ModelEma, create_exp_dir
from OFA_mbv3_extended.my_data_loaders.my_random_resize_crop import MyModRandomResizedCrop
from OFA_mbv3_extended.networks.nets.my_networks import get_valid_nw_keys
from .schedulers import CosineLRScheduler

try:
    from apex import amp

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass


class Trainer(ABC):
    def __init__(
            self,
            model,  # the network model to be trained
            # below are training configurations
            n_epochs,  # number of training epochs
            lr_init,  # initial learning rate
            data_provider,  # data provider including both train and valid data loader
            optimizer="sgd",  # name of the optimizer
            patience=8,
            cur_epoch=0,  # current epoch from which to train
            lr_end=0.,  # final learning rate
            lr_warmup=0.0001,  # warmup learning rate
            lr_warmup_epochs=5,  # number of epochs to warm-up learning rate
            momentum=0.9,
            wd=3e-4,  # weight decay
            grad_clip=5,  # gradient clipping
            model_ema=False,  # keep track of moving average of model parameters
            model_ema_decay=0.9998,  # decay factor for model weights moving average
            logger=None,  # logger handle for recording
            save_path='.tmp'  # path to save experiment data
    ):

        self.cur_epoch = cur_epoch
        self.n_epochs = n_epochs
        self.grad_clip = grad_clip
        self.patience = patience

        self.data_provider = data_provider

        if optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD(model.parameters(), lr_init, momentum=momentum, weight_decay=wd)
        elif optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr_init, weight_decay=wd)
        else:
            raise NotImplementedError("specified optimizer not implemented ")

        self.criterion = torch.nn.CrossEntropyLoss()

        lr_cycle_args = {'cycle_mul': 1., 'cycle_decay': 0.1, 'cycle_limit': 1}
        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=n_epochs,
            lr_min=lr_end,
            warmup_lr_init=lr_warmup,
            warmup_t=lr_warmup_epochs,
            k_decay=1.0,
            **lr_cycle_args,
        )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, n_epochs, eta_min=lr_end)

        self.model = model

        if model_ema:
            self.model_ema = ModelEma(model, decay=model_ema_decay)
        else:
            self.model_ema = model_ema

        self.save_path = save_path
        create_exp_dir(self.save_path)  # create an experiment folder

        # setup progress logger
        if logger is None:
            import sys
            import logging

            log_format = '%(asctime)s %(message)s'
            logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
            fh = logging.FileHandler(os.path.join(self.save_path, 'log.txt'))
            fh.setFormatter(logging.Formatter(log_format))
            self.logger = logging.getLogger()
            self.logger.addHandler(fh)
        else:
            self.logger = logger

        # resolve AMP arguments based on PyTorch / Apex availability
        use_amp = None
        if has_apex:
            use_amp = 'apex'
        elif has_native_amp:
            use_amp = 'native'
        else:
            print("Neither APEX or native Torch AMP is available, using float32. "
                  "Install NVIDA apex or upgrade to PyTorch 1.6")

        # setup automatic mixed-precision (AMP) loss scaling and op casting
        self.amp_autocast = suppress  # do nothing
        self.loss_scaler = None
        if use_amp == 'apex':
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')
            self.loss_scaler = ApexScaler()
            self.logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
        elif use_amp == 'native':
            self.amp_autocast = torch.cuda.amp.autocast
            self.loss_scaler = NativeScaler()
            self.logger.info('Using native Torch AMP. Training in mixed precision.')
        else:
            self.logger.info('AMP not enabled. Training in float32.')

    def train_one_epoch(self, epoch, run_str='', no_logs=False):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def validate(self, epoch):
        raise NotImplementedError


class TrainerSE(Trainer):
    def __init__(
            self,
            model,
            n_epochs,
            lr_init,
            data_provider,
            optimizer="sgd",
            patience=8,
            cur_epoch=0,
            lr_end=0.,
            lr_warmup=0.0001,
            lr_warmup_epochs=5,
            momentum=0.9,
            wd=3e-4,
            grad_clip=5,
            model_ema=False,
            model_ema_decay=0.9998,
            logger=None,
            save_path='.tmp'
    ):

        super(TrainerSE, self).__init__(
            model,
            n_epochs,
            lr_init,
            data_provider,
            optimizer,
            patience,
            cur_epoch,
            lr_end,
            lr_warmup,
            lr_warmup_epochs,
            momentum,
            wd,
            grad_clip,
            model_ema,
            model_ema_decay,
            logger,
            save_path
        )

    def train_one_epoch(self, epoch, run_str='', no_logs=False):
        self.model.train()

        MyModRandomResizedCrop.EPOCH = epoch  # such that sampled image resolution is changing
        n_batch = len(self.data_provider.train)
        losses = AverageMeter()
        metric_dict = get_metric_dict()

        with tqdm(
            total=n_batch,
            desc='Training Epoch #{} {}'.format(epoch + 1, run_str),
            disable=no_logs,
        ) as t:
            num_updates = epoch * n_batch
            for step, (x, target) in enumerate(self.data_provider.train):

                MyModRandomResizedCrop.BATCH = step
                device = get_net_device(self.model)
                x, target = x.to(device), target.to(device)

                with self.amp_autocast():
                    logits = self.model(x)
                    loss = self.criterion(logits, target)

                losses.update(loss.item(), x.size(0))
                update_metric(metric_dict, logits, target)

                self.optimizer.zero_grad()
                if self.loss_scaler is not None:
                    self.loss_scaler(
                        loss,
                        self.optimizer,
                        clip_grad=self.grad_clip,
                        clip_mode='norm',
                        parameters=self.model.parameters(),
                        create_graph=False
                    )
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                if self.model_ema:
                    self.model_ema.update(self.model)

                torch.cuda.synchronize()
                num_updates += 1

                lrl = [param_group['lr'] for param_group in self.optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                self.scheduler.step_update(num_updates=num_updates, metric=losses.avg)

                t.set_postfix({
                    'loss': losses.avg,
                    **get_metric_vals(metric_dict, return_dict=True),
                    'lr': lr,
                    'img_size': x.size(2),
                })
                t.update(1)

        return losses.avg, get_metric_vals(metric_dict)

    def validate(self, epoch, is_test=False):
        data_loader = self.data_provider.valid if not is_test else self.data_provider.test

        image_sizes = self.data_provider.image_size
        if not isinstance(image_sizes, list):
            image_sizes = list(image_sizes)

        losses, top1s, top5s = [], [], []
        for img_size in image_sizes:
            # set image size
            self.data_provider.assign_active_img_size(img_size)

            # measure acc
            loss, (top1, top5) = validate_se(self.model, data_loader, self.criterion, epoch=epoch)
            losses.append(loss)
            top1s.append(top1)
            top5s.append(top5)

        return np.mean(losses), (np.mean(top1s), np.mean(top5s))

    def train(self):

        #early_stopping_meter = AccEarlyStoppingMeter(patience=self.patience) #esm
        best_acc = None

        for epoch in range(self.cur_epoch, self.n_epochs):
            # self.logger.info('epoch {:d} lr {:.2e}'.format(epoch, self.scheduler.get_lr()[0]))
            self.logger.info('epoch {:d} lr {:.2e}'.format(epoch + 1, self.scheduler.get_epoch_values(epoch)[0]))

            train_loss, (train_top1, train_top5) = self.train_one_epoch(epoch)

            print_str = "TRAIN: loss={:.4f}, top1={:.4f}, top5={:.4f} -- ".format(train_loss, train_top1, train_top5)

            save_dict = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }

            # # uncomment below for debugging, cost additional 20 secs per epoch
            # if self.model_ema:
            #     valid_loss_ema, (valid_acc_ema, _) = validate(
            #         self.model_ema.ema,
            #         self.valid_dataloader,
            #         self.criterion, epoch=epoch
            #     )
            #     print_str += ", valid acc ema = {:.4f}".format(valid_acc_ema)
            #     save_dict['state_dict_ema'] = get_state_dict(self.model_ema)

            # validate every epoch
            valid_loss, (valid_acc1, valid_acc5) = self.validate(epoch)
            print_str += "VALID: loss={:.4f}, top1={:.4f}, top5={:.4f}".format(valid_loss, valid_acc1, valid_acc5)

            self.logger.info(print_str)
            self.scheduler.step(epoch + 1)
            torch.save(save_dict, os.path.join(self.save_path, 'checkpoint.pth.tar'))

            if best_acc is None or valid_acc1 > best_acc:
                best_acc = valid_acc1
                torch.save(save_dict, os.path.join(self.save_path, 'model_best.pth.tar'))

            # should_stop = early_stopping_meter.update(valid_acc1) #esm
            # if should_stop: #esm
            #    return #esm


class TrainerEE(Trainer):
    def __init__(
            self,
            model,
            n_epochs,
            lr_init,
            data_provider,
            branches_weights,
            ensemble_weights,
            optimizer="sgd",
            patience=8,
            cur_epoch=0,
            lr_end=0.,
            lr_warmup=0.0001,
            lr_warmup_epochs=5,
            momentum=0.9,
            wd=3e-4,
            grad_clip=5,
            model_ema=False,
            model_ema_decay=0.9998,
            logger=None,
            save_path='.tmp'
    ):
        super(TrainerEE, self).__init__(
            model,
            n_epochs,
            lr_init,
            data_provider,
            optimizer,
            patience,
            cur_epoch,
            lr_end,
            lr_warmup,
            lr_warmup_epochs,
            momentum,
            wd,
            grad_clip,
            model_ema,
            model_ema_decay,
            logger,
            save_path,
        )

        self.branches_weights = branches_weights
        self.ensemble_weights = ensemble_weights

    def train_one_epoch(self, epoch, run_str='', no_logs=False):
        self.model.train()

        MyModRandomResizedCrop.EPOCH = epoch  # such that sampled image resolution is changing
        n_batch = len(self.data_provider.train)
        net_loss_meter = AverageMeter()
        net_metric_dict = get_metric_dict()

        branches_loss_meter = []
        branches_metric_dict = []
        for _ in range(5):
            branches_loss_meter.append(AverageMeter())
            branches_metric_dict.append(get_metric_dict())

        with tqdm(
            total=n_batch,
            desc='Training Epoch #{} {}'.format(epoch + 1, run_str),
            disable=no_logs,
        ) as t:

            num_updates = epoch * n_batch
            for step, (x, target) in enumerate(self.data_provider.train):

                MyModRandomResizedCrop.BATCH = step
                device = get_net_device(self.model)
                x, target = x.to(device), target.to(device)

                with self.amp_autocast():
                    outputs = self.model(x)
                    branches_losses = [self.criterion(output, target) for output in outputs]

                    # compute weighted loss for each branch
                    weighted_losses = []
                    for branch_loss, branch_weight in zip(branches_losses, self.branches_weights):
                        weighted_losses.append(branch_loss * branch_weight)

                    # add the losses together
                    net_loss = 0
                    for weighted_loss in weighted_losses:
                        net_loss += weighted_loss

                    weighted_outputs = []
                    for output, weight in zip(outputs, self.ensemble_weights):
                        weighted_outputs.append(output * weight)

                    # use the weighted outputs and sum to obtain the net output
                    net_output = torch.stack(weighted_outputs)
                    net_output = torch.sum(net_output, dim=0)

                # update statistics
                net_loss_meter.update(net_loss.item(), x.size(0))
                update_metric(net_metric_dict, net_output, target)

                for branch_loss_meter, weighted_loss in zip(branches_loss_meter, weighted_losses):
                    branch_loss_meter.update(weighted_loss.item(), x.size(0))

                for branch_metric_dict, output in zip(branches_metric_dict, outputs):
                    update_metric(branch_metric_dict, output, target)

                # optimization step, based on net loss
                self.optimizer.zero_grad()
                if self.loss_scaler is not None:
                    self.loss_scaler(
                        net_loss,
                        self.optimizer,
                        clip_grad=self.grad_clip,
                        clip_mode='norm',
                        parameters=self.model.parameters(),
                        create_graph=False
                    )
                else:
                    net_loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                if self.model_ema:
                    self.model_ema.update(self.model)

                torch.cuda.synchronize()
                num_updates += 1

                lrl = [param_group['lr'] for param_group in self.optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                self.scheduler.step_update(num_updates=num_updates, metric=net_loss_meter.avg)

                t.set_postfix({
                    'net_loss': net_loss_meter.avg,
                    **get_metric_vals(net_metric_dict, return_dict=True),
                    'lr': lr,
                    'img_size': x.size(2),
                })
                t.update(1)

        # compute results to return
        net_loss_avg = net_loss_meter.avg
        branches_loss_avg = [branch_loss_meter.avg for branch_loss_meter in branches_loss_meter]
        net_metric_vals_list = get_metric_vals(net_metric_dict, return_dict=False)

        branches_metric_vals_lists = [get_metric_vals(branch_metric_dict, return_dict=False)
                                      for branch_metric_dict in branches_metric_dict]
        top1s = [top[0] for top in branches_metric_vals_lists]
        top5s = [top[1] for top in branches_metric_vals_lists]
        branches_metric_vals_list = [top1s, top5s]

        return net_loss_avg, net_metric_vals_list, branches_loss_avg, branches_metric_vals_list

    def train(self):

        # early_stopping_meter = AccEarlyStoppingMeter(patience=self.patience) #esm
        best_acc = None

        for epoch in range(self.cur_epoch, self.n_epochs):
            # self.logger.info('epoch {:d} lr {:.2e}'.format(epoch, self.scheduler.get_lr()[0]))
            self.logger.info('\nepoch {:d} lr {:.2e}'.format(epoch + 1, self.scheduler.get_epoch_values(epoch)[0]))

            tr_net_loss, (tr_net_top1, tr_net_top5), tr_br_loss, (tr_br_top1, tr_br_top5) = self.train_one_epoch(epoch)
            tr_br_loss_str = "["+", ".join([str(s) for s in [round(elem, 4) for elem in tr_br_loss]])+"]"
            tr_br_top1_str = "["+", ".join([str(s) for s in [round(elem, 4) for elem in tr_br_top1]])+"]"
            tr_br_top5_str = "["+", ".join([str(s) for s in [round(elem, 4) for elem in tr_br_top5]])+"]"

            print_str = "\nTRAIN: "
            print_str += "\nnet_loss={:.4f}, net_top1={:.4f}, net top5={:.4f} ".format(tr_net_loss, tr_net_top1, tr_net_top5)
            print_str += "\nbr_loss=" + tr_br_loss_str + "\nbr_top1=" + tr_br_top1_str + "\nbr_top5=" + tr_br_top5_str

            save_dict = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }

            # validate every epoch
            v_net_loss, (v_net_top1, v_net_top5), v_br_loss, (v_br_top1, v_br_top5) = self.validate(epoch)
            v_br_loss_str = "[" + ", ".join([str(s) for s in [round(elem, 4) for elem in v_br_loss]]) + "]"
            v_br_top1_str = "[" + ", ".join([str(s) for s in [round(elem, 4) for elem in v_br_top1]]) + "]"
            v_br_top5_str = "[" + ", ".join([str(s) for s in [round(elem, 4) for elem in v_br_top5]]) + "]"

            print_str += "\nVALID: "
            print_str += "\nnet_loss={:.4f}, net_top1={:.4f}, net top5={:.4f} ".format(v_net_loss, v_net_top1, v_net_top5)
            print_str += "\nbr_loss=" + v_br_loss_str + "\nbr_top1=" + v_br_top1_str + "\nbr_top5=" + v_br_top5_str + "\n"

            self.logger.info(print_str)
            self.scheduler.step(epoch + 1)
            torch.save(save_dict, os.path.join(self.save_path, 'checkpoint.pth.tar'))

            if best_acc is None or v_net_top1 > best_acc:
                best_acc = v_net_top1
                torch.save(save_dict, os.path.join(self.save_path, 'model_best.pth.tar'))

            # should_stop = early_stopping_meter.update(v_net_top1) #esm
            # if should_stop: #esm
            #    return #esm

    def validate(self, epoch, is_test=False):
        data_loader = self.data_provider.valid if not is_test else self.data_provider.test

        image_sizes = self.data_provider.image_size
        if not isinstance(image_sizes, list):
            image_sizes = list(image_sizes)

        net_losses, net_top1s, net_top5s = [], [], []
        branches_losses, branches_top1s, branches_top5s = [], [], []

        for img_size in image_sizes:
            # set image size
            self.data_provider.assign_active_img_size(img_size)

            # measure acc
            net_loss, (net_top1, net_top5), brs_loss, (brs_top1s, brs_top5s) = validate_ee(
                net=self.model,
                data_loader=data_loader,
                criterion=self.criterion,
                branches_weights=self.branches_weights,
                ensemble_weights=self.ensemble_weights,
                epoch=epoch
            )
            net_losses.append(net_loss)
            net_top1s.append(net_top1)
            net_top5s.append(net_top5)
            branches_losses.append(brs_loss)
            branches_top1s.append(brs_top1s)
            branches_top5s.append(brs_top5s)

        mean_net_losses = np.mean(net_losses)   # value
        mean_net_top1s = np.mean(net_top1s)     # value
        mean_net_top5s = np.mean(net_top5s)     # value
        mean_branches_losses = np.mean(branches_losses, axis=0)     # array
        mean_branches_top1s = np.mean(branches_top1s, axis=0)   # array
        mean_branches_top5s = np.mean(branches_top5s, axis=0)   # array

        return (
            mean_net_losses,
            (mean_net_top1s, mean_net_top5s),
            mean_branches_losses,
            (mean_branches_top1s, mean_branches_top5s)
        )


class SuperNetTrainer(ABC):
    def __init__(
            self,
            supernet,  # the supernet model to be trained
            teacher_model,  # we use the full capacity supernet to supervise the subnet training
            search_space,  # the search space from which a subpart of supernet can be sampled
            # below are training configurations
            n_epochs,  # number of training epochs
            lr_init,  # initial learning rate
            data_provider,  # data provider including both train and valid data loader
            cur_epoch=0,  # current epoch from which to train
            lr_end=0.,  # final learning rate
            lr_warmup=0.0001,  # warmup learning rate
            lr_warmup_epochs=5,  # number of epochs to warm-up learning rate
            momentum=0.9,
            wd=3e-4,  # weight decay
            grad_clip=5,  # gradient clipping
            supernet_ema=False,  # keep track of moving average of model parameters
            supernet_ema_decay=0.9998,  # decay factor for model weights moving average
            logger=None,  # logger handle for recording
            save_path='.tmp',  # path to save experiment data
            # additional arguments for training supernet
            dynamic_batch_size=1,  # number of architectures to accumulate gradient
            kd_ratio=1.0,  # teacher-student knowledge distillation hyperparameter
            sub_train_size=2000,  # number of images to calibrate BN stats
            sub_train_batch_size=200,  # batch size for subset train dataloader
            distributions=None,  # instead of uniform sampling from search space,
            # we can sample explicitly from a distribution
            report_freq=10,  # frequency to run validation and print results
            # -- below newly added values
            optim="sgd",
            patience=8,  # early stopping patience parameter
            n_max_val=32,  # max number of subnets to evaluate during validation phase
            is_se=True,  # is single-exit or early-exit network
            branches_weights=None,
            ensemble_weights=None
    ):

        self.cur_epoch = cur_epoch
        self.n_epochs = n_epochs
        self.grad_clip = grad_clip
        self.report_freq = report_freq

        self.data_provider = data_provider
        self.sub_train_size = sub_train_size
        self.sub_train_batch_size = sub_train_batch_size

        self.distributions = distributions

        # new attributes
        self.patience = patience
        self.n_max_val = n_max_val
        self.is_se = is_se
        self.branches_weights = branches_weights
        self.ensemble_weights = ensemble_weights

        # choose the optimizer
        if optim.lower() == "sgd":
            self.optimizers = [torch.optim.SGD(model.parameters(), lr_init, momentum=momentum, weight_decay=wd)
                               for model in supernet.engine]  # we have one optimizer for supernet of each width_mult
        elif optim.lower() == "adam":
            self.optimizers = [torch.optim.Adam(model.parameters(), lr_init, weight_decay=wd)
                               for model in supernet.engine]  # we have one optimizer for supernet of each width_mult
        else:
            raise ValueError

        lr_cycle_args = {'cycle_mul': 1., 'cycle_decay': 0.1, 'cycle_limit': 1}
        self.schedulers = [CosineLRScheduler(
            optimizer=optimizer,
            t_initial=n_epochs,
            lr_min=lr_end,
            warmup_lr_init=lr_warmup,
            warmup_t=lr_warmup_epochs,
            k_decay=1.0,
            **lr_cycle_args,
        ) for optimizer in self.optimizers]  # we have one lr scheduler for supernet of each wid_mult

        # self.schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=lr_end)
        #                    for optimizer in self.optimizers]  # we have one lr scheduler for supernet of each wid_mult

        self.criterion = torch.nn.CrossEntropyLoss()

        self.supernet = supernet

        if supernet_ema:
            self.supernet_ema = [ModelEma(model, decay=supernet_ema_decay) for model in self.supernet.engine]
        else:
            self.supernet_ema = supernet_ema

        self.teacher_model = teacher_model
        self.search_space = search_space
        self.dynamic_batch_size = dynamic_batch_size
        self.kd_ratio = kd_ratio

        self.save_path = save_path
        create_exp_dir(self.save_path)  # create an experiment folder

        # setup progress logger
        if logger is None:
            import sys
            import logging

            log_format = '%(asctime)s %(message)s'
            logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
            fh = logging.FileHandler(os.path.join(self.save_path, 'log.txt'))
            fh.setFormatter(logging.Formatter(log_format))
            self.logger = logging.getLogger()
            self.logger.addHandler(fh)
        else:
            self.logger = logger

        # # resolve AMP arguments based on PyTorch / Apex availability
        # use_amp = None
        # if has_apex:
        #     use_amp = 'apex'
        # elif has_native_amp:
        #     use_amp = 'native'
        # else:
        #     print("Neither APEX or native Torch AMP is available, using float32. "
        #           "Install NVIDA apex or upgrade to PyTorch 1.6")

        # # setup automatic mixed-precision (AMP) loss scaling and op casting
        # self.amp_autocast = suppress  # do nothing
        # self.loss_scaler = None
        # if use_amp == 'apex':
        #     for i in range(len(self.optimizers)):
        #         self.supernet.engine[i], self.optimizers[i] = amp.initialize(
        #             self.supernet.engine[i], self.optimizers[i], opt_level='O1')
        #     # self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')
        #     self.loss_scalers = [ApexScaler() for _ in range(len(self.optimizers))]
        #     self.logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
        # elif use_amp == 'native':
        #     self.amp_autocast = torch.cuda.amp.autocast
        #     self.loss_scalers = [NativeScaler() for _ in range(len(self.optimizers))]
        #     # self.loss_scaler = NativeScaler()
        #     self.logger.info('Using native Torch AMP. Training in mixed precision.')
        # else:
        #     self.logger.info('AMP not enabled. Training in float32.')

    def train_one_epoch(self, epoch, run_str='', no_logs=False):

        # switch to training mode
        for model in self.supernet.engine:
            model.train()

        MyModRandomResizedCrop.EPOCH = epoch  # such that sampled image resolution is changing
        n_batch = len(self.data_provider.train)

        losses = AverageMeter()
        metric_dict = get_metric_dict()

        with tqdm(
                total=n_batch,
                desc='Training Epoch #{} {}'.format(epoch + 1, run_str),
                disable=no_logs,
        ) as t:

            num_updates = epoch * n_batch
            for step, (x, target) in enumerate(self.data_provider.train):

                MyModRandomResizedCrop.BATCH = step
                x, target = x.cuda(), target.cuda(non_blocking=True)

                # soft target
                with torch.no_grad():
                    if self.is_se:
                        soft_logits = self.teacher_model(x).detach()
                        soft_label = F.softmax(soft_logits, dim=1)
                    else:
                        soft_logits_list = self.teacher_model(x)
                        for soft_logits in soft_logits_list:
                            soft_logits.detach()

                        # weight the logits
                        weighted_logits = []
                        for soft_logits, weight in zip(soft_logits_list, self.ensemble_weights):
                            weighted_logits.append(soft_logits * weight)

                        teacher_soft_logits = torch.stack(weighted_logits)
                        teacher_soft_logits = torch.sum(teacher_soft_logits, dim=0)

                        soft_label = F.softmax(teacher_soft_logits, dim=1)

                # clean gradients
                for model in self.supernet.engine:
                    model.zero_grad()
                for optimizer in self.optimizers:
                    optimizer.zero_grad()

                # sample architectures from search space
                subnets = self.search_space.sample(self.dynamic_batch_size, distributions=self.distributions)

                subnet_str, loss_of_subnets = '', []
                for _, subnet_settings in enumerate(subnets):
                    # set random seed before sampling
                    subnet_seed = int('%d%.3d%.3d' % (epoch * n_batch + step, _, 0))
                    random.seed(subnet_seed)
                    subnet_settings.pop('r')  # remove image size key
                    self.supernet.set_active_subnet(**subnet_settings)
                    subnet_str += '%d: ' % _ + ','.join(['%s_%s' % (
                        key, '%.1f' % subset_mean(val, 0) if isinstance(val, list) else val
                    ) for key, val in subnet_settings.items()]) + ' || '

                    # with self.amp_autocast():
                    logits = self.supernet.forward(x)

                    # same line but soft_labels changes depending on if SE or EE network
                    kd_loss = cross_entropy_loss_with_soft_target(logits, soft_label)

                    loss = self.kd_ratio * kd_loss + self.criterion(logits, target)
                    loss = loss * (2 / (self.kd_ratio + 1))

                    loss_of_subnets.append(loss.item())
                    update_metric(metric_dict, logits, target)

                    loss.backward()
                    nn.utils.clip_grad_norm_(self.supernet.parameters(), self.grad_clip)

                for optimizer in self.optimizers:
                    optimizer.step()

                if self.supernet_ema:
                    for model, model_ema in zip(self.supernet.engine, self.supernet_ema):
                        model_ema.update(model)

                torch.cuda.synchronize()
                num_updates += 1
                losses.update(list_mean(loss_of_subnets), x.size(0))

                lrl = [param_group['lr'] for param_group in self.optimizers[0].param_groups]
                lr = sum(lrl) / len(lrl)

                for scheduler in self.schedulers:
                    scheduler.step_update(num_updates=num_updates, metric=losses.avg)

                t.set_postfix({
                    'loss': losses.avg,
                    **get_metric_vals(metric_dict, return_dict=True),
                    'lr': lr,
                    'img_size': x.size(2),
                    'str': subnet_str,
                })
                t.update(1)

        return losses.avg, get_metric_vals(metric_dict)

    def validate(self, epoch, is_test=False):

        image_sizes = self.data_provider.image_size
        if not isinstance(image_sizes, list):
            image_sizes = list(image_sizes)

        # create the combinations to validate performance
        width_mult = [min(self.supernet.width_mult_list), max(self.supernet.width_mult_list)]
        ks = [min(self.supernet.ks_list), max(self.supernet.ks_list)]
        depth = [min(self.supernet.depth_list), max(self.supernet.depth_list)]
        expand = [min(self.supernet.expand_ratio_list), max(self.supernet.expand_ratio_list)]
        net_depth = [min(self.supernet.net_depth_list), max(self.supernet.net_depth_list)] \
            if hasattr(self.supernet, "net_depth_list") else [-1]
        net_width = [max(get_valid_nw_keys(min(self.supernet.net_width_list))), max(get_valid_nw_keys(max(self.supernet.net_width_list)))] \
            if hasattr(self.supernet, "net_width_list") else [-1]
        # [4,7] instead of [1,7] so that on only one layer active, the one active (4=100) is the inverted bottleneck

        val_settings = list(itertools.product(image_sizes, width_mult, ks, depth, expand, net_depth, net_width))

        # limit the amount of validated subnets
        if 0 < self.n_max_val < len(val_settings):
            val_settings = random.sample(val_settings, self.n_max_val)

        # order tuples by values of (wm, img_size, nd, nw, ks, d, e) for evaluation
        # DOES NOT change structure of tuples, still (r, w, ks, d, e, nd, nw)
        val_settings = sorted(val_settings, key=lambda tup: (tup[1], tup[0], tup[5], tup[6], tup[2], tup[3], tup[4]))

        print(val_settings)
        losses, top1s, top5s = [], [], []
        for r, w, ks, d, e, nd, nw in val_settings:

            settings_str = 'r={:d}, w={:.1f}, ks={:d}, d={:d}, e={:d}'.format(r, w, ks, d, e)
            settings = {
                "r": r,
                "w": w,
                "ks": ks,
                "e": e,
                "d": d,
            }

            if nd != -1:
                settings_str += ", nd={:d}".format(nd)
                settings["nd"] = nd

            if nw != -1:
                settings_str += ", nw={:d}".format(nw)
                settings["nw"] = nw

            self.logger.info(settings_str)

            # set image size
            self.data_provider.assign_active_img_size(settings.pop("r"))
            dl = self.data_provider.valid if not is_test else self.data_provider.test
            sdl = self.data_provider.build_sub_train_loader(self.sub_train_size, self.sub_train_batch_size)

            # set subnet settings
            self.supernet.set_active_subnet(**settings)
            subnet = self.supernet.get_active_subnet(preserve_weight=True)

            # reset BN running statistics
            subnet.train()
            set_running_statistics(subnet, sdl)

            # measure acc
            loss, (top1, top5) = validate_se(subnet, dl, self.criterion, epoch=epoch)
            losses.append(loss)
            top1s.append(top1)
            top5s.append(top5)

            self.logger.info(f"Subnet top1: {top1:.2f}")

        return np.mean(losses), (np.mean(top1s), np.mean(top5s))

    def train(self):

        # early_stopping_meter = AccEarlyStoppingMeter(patience=self.patience) #esm
        # should_stop = False #esm
        best_acc = None

        for epoch in range(self.cur_epoch, self.n_epochs):

            # self.logger.info('epoch {:d} lr {:.2e}'.format(epoch, self.schedulers[0].get_lr()[0]))
            self.logger.info('epoch {:d} lr {:.2e}'.format(epoch + 1, self.schedulers[0].get_epoch_values(epoch)[0]))

            train_loss, (train_top1, train_top5) = self.train_one_epoch(epoch)

            print_str = "TRAIN: loss={:.4f}, top1={:.4f}, top5={:.4f} -- ".format(train_loss, train_top1, train_top5)

            save_dict = {
                'epoch': epoch,
                'model_w1.0_state_dict': self.supernet.engine[0].state_dict(),
                'model_w1.2_state_dict': self.supernet.engine[1].state_dict(),
                'optimizer_w1.0_state_dict': self.optimizers[0].state_dict(),
                'optimizer_w1.2_state_dict': self.optimizers[1].state_dict()
            }

            torch.save(save_dict, os.path.join(self.save_path, 'checkpoint.pth.tar'))

            if (epoch + 1) % self.report_freq == 0:

                valid_loss, (valid_acc1, valid_acc5) = self.validate(epoch)
                print_str += "VALID: loss={:.4f}, top1={:.4f}, top5={:.4f}".format(valid_loss, valid_acc1, valid_acc5)

                if best_acc is None or valid_acc1 > best_acc:
                    best_acc = valid_acc1
                    torch.save(save_dict, os.path.join(self.save_path, 'model_best.pth.tar'))

                # should_stop = early_stopping_meter.update(valid_acc1) #esm

            self.logger.info(print_str)

            for scheduler in self.schedulers:
                scheduler.step(epoch + 1)

            # if should_stop: #esm
            #    return #esm

