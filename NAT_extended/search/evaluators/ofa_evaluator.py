# import sys
# sys.path.insert(0, './')

import copy
import time
import warnings
from abc import ABC

import torch
import torch.backends.cudnn as cudnn
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics
from ofa.utils import get_net_device
from torchprofile import profile_macs
from tqdm import tqdm

from NAT_extended.evaluation.evaluate import validate_se


class OFAEvaluator(ABC):
    def __init__(
         self,
         supernet,  # supernet generator class
         data_provider,  # data provider class
         sub_train_size,  # number of images to calibrate BN stats
         sub_train_batch_size,  # batch size for subset train dataloader
     ):

        self.supernet = supernet
        self.data_provider = data_provider
        self.num_classes = data_provider.n_classes
        self.sub_train_size = sub_train_size
        self.sub_train_batch_size = sub_train_batch_size

        self.criterion = torch.nn.CrossEntropyLoss()

    @staticmethod
    def _calc_params(subnet):
        return sum(p.numel() for p in subnet.parameters() if p.requires_grad) / 1e6  # in unit of Million

    @staticmethod
    def _calc_flops(subnet, dummy_data):
        dummy_data = dummy_data.to(get_net_device(subnet))
        return profile_macs(subnet, dummy_data) / 1e6  # in unit of MFLOPs

    @staticmethod
    def measure_latency(subnet, input_size, iterations=None, verbose=False):
        """ Be aware that latency will fluctuate depending on the hardware operating condition,
        e.g., loading, temperature, etc. """

        if verbose:
            print("measuring latency....")

        cudnn.enabled = True
        cudnn.benchmark = True

        subnet.eval()
        model = subnet.cuda()
        input = torch.randn(*input_size).cuda()

        with torch.no_grad():
            for _ in range(100):
                model(input)

            if iterations is None:
                elapsed_time = 0
                iterations = 100
                while elapsed_time < 1:
                    torch.cuda.synchronize()
                    torch.cuda.synchronize()
                    t_start = time.time()
                    for _ in range(iterations):
                        model(input)
                    torch.cuda.synchronize()
                    torch.cuda.synchronize()
                    elapsed_time = time.time() - t_start
                    iterations *= 2
                FPS = iterations / elapsed_time
                iterations = int(FPS * 6)

            if verbose:
                print('=========Speed Testing=========')

            torch.cuda.synchronize()
            torch.cuda.synchronize()
            t_start = time.time()
            if verbose:
                for _ in tqdm(range(iterations)):
                    model(input)
            else:
                for _ in range(iterations):
                    model(input)
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            elapsed_time = time.time() - t_start
            latency = elapsed_time / iterations * 1000
        torch.cuda.empty_cache()
        # FPS = 1000 / latency (in ms)
        return latency

    def _measure_latency(self, subnet, input_size):
        # return self.measure_latency(subnet, input_size)   # >>> original, takes way too much time during search

        iterations = 100
        return self.measure_latency(subnet, input_size, iterations)

    @staticmethod
    def eval_acc(subnet, dl, sdl, criterion):

        # reset BN running statistics
        subnet.train()
        set_running_statistics(subnet, sdl)
        # measure acc (note: all subnets are single-exit)
        loss, (top1, top5) = validate_se(subnet, dl, criterion)
        return loss, top1, top5

    """ high-fidelity evaluation by inference on validation data """
    def evaluate(self, _subnets, objs='acc&flops&params&latency', print_progress=True):

        subnets = copy.deepcopy(_subnets)  # make a copy of the archs to be evaluated
        batch_stats = []

        for i, subnet_str in enumerate(subnets):
            if print_progress:
                print("evaluating subnet {}:".format(i + 1))
                print(subnet_str)

            stats = {}
            # set subnet accordingly
            image_scale = subnet_str.pop('r')
            input_size = (1, 3, image_scale, image_scale)

            # create dummy data for measuring flops
            dummy_data = torch.rand(*input_size)

            self.supernet.set_active_subnet(**subnet_str)
            subnet = self.supernet.get_active_subnet(preserve_weight=True)
            subnet.cuda()

            print_str = ''
            if 'acc' in objs:
                # set the image scale
                self.data_provider.assign_active_img_size(image_scale)
                dl = self.data_provider.valid
                sdl = self.data_provider.build_sub_train_loader(self.sub_train_size, self.sub_train_batch_size)

                # compute top-1 accuracy
                _, top1, _ = self.eval_acc(subnet, dl, sdl, self.criterion)

                # batch_acc.append(top1)
                stats['acc'] = top1
                print_str += 'Top1 = {:.2f}'.format(top1)

            # calculate #params and #flops
            if 'params' in objs:
                params = self._calc_params(subnet)
                # batch_params.append(params)
                stats['params'] = params
                print_str += ', #Params = {:.2f}M'.format(params)

            if 'flops' in objs:
                with warnings.catch_warnings():  # ignore warnings, use w/ caution
                    warnings.simplefilter("ignore")
                    flops = self._calc_flops(subnet, dummy_data)
                # batch_flops.append(flops)
                stats['flops'] = flops
                print_str += ', #FLOPs = {:.2f}M'.format(flops)

            if 'latency' in objs:
                latency = self._measure_latency(subnet, input_size)
                # batch_latency.append(latency)
                stats['latency'] = latency
                print_str += ', FPS = {:d}'.format(int(1000 / latency))

            if print_progress:
                print(print_str)
            batch_stats.append(stats)

        return batch_stats

