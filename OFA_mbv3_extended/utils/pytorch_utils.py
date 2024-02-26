import copy
import time

import torch
import torch.nn as nn
from ofa.utils import count_parameters, rm_bn_from_net, get_net_device


def module_require_grad(module):
    return module.parameters().__next__().requires_grad


def count_net_flops(net, data_shape=(1, 3, 224, 224)):

    if isinstance(net, nn.DataParallel):
        net = net.module

    # ALTERNATIVE 1: original
    # from .flops_counter import profile
    # flop, _ = profile(copy.deepcopy(net), data_shape)

    # ALTERNATIVE 2: torchprofile.profile_macs
    import warnings
    from torchprofile import profile_macs
    with warnings.catch_warnings():  # ignore warnings, use w/ caution
        warnings.simplefilter("ignore")
        dummy_data = torch.rand(*data_shape)
        dummy_data = dummy_data.to(get_net_device(net))
        flop = profile_macs(copy.deepcopy(net), dummy_data)

    # ALTERNATIVE 3: fvcore (note: requires both fvcores and iopath)
    # from fvcore.nn import FlopCountAnalysis
    # dummy_data = torch.rand(*data_shape)
    # fca = FlopCountAnalysis(copy.deepcopy(net), dummy_data)
    # flop = fca.total()

    return flop


# use NAT function for synchronization on cuda
def measure_net_latency(net, l_type='gpu8', fast=True, input_shape=(3, 224, 224), clean=False):
    if isinstance(net, nn.DataParallel):
        net = net.module

    # remove bn from graph
    rm_bn_from_net(net)

    # return `ms`
    if 'gpu' in l_type:
        l_type, batch_size = l_type[:3], int(l_type[3:])
    else:
        batch_size = 1

    data_shape = [batch_size] + list(input_shape)
    if l_type == 'cpu':
        if fast:
            n_warmup = 5
            n_sample = 10
        else:
            n_warmup = 50
            n_sample = 50
        if get_net_device(net) != torch.device('cpu'):
            if not clean:
                print('move net to cpu for measuring cpu latency')
            net = copy.deepcopy(net).cpu()
    elif l_type == 'gpu':
        if fast:
            n_warmup = 5
            n_sample = 10
        else:
            n_warmup = 50
            n_sample = 50
    else:
        raise NotImplementedError
    images = torch.zeros(data_shape, device=get_net_device(net))

    measured_latency = {'warmup': [], 'sample': []}
    net.eval()
    with torch.no_grad():
        for i in range(n_warmup):
            # ensure that context initialization and normal_() operations
            # finish before you start measuring time
            torch.cuda.synchronize()
            # inner_start_time = time.time()
            inner_start_time = time.perf_counter()
            net(images)
            torch.cuda.synchronize()  # wait for inference to finish
            used_time = (time.perf_counter() - inner_start_time) * 1e3  # ms
            measured_latency['warmup'].append(used_time)
            if not clean:
                print('Warmup %d: %.3f' % (i, used_time))
        outer_start_time = time.time()
        for i in range(n_sample):
            net(images)
        total_time = (time.time() - outer_start_time) * 1e3  # ms
        measured_latency['sample'].append((total_time, n_sample))
    return total_time / n_sample, measured_latency


def get_net_info(net, input_shape=(3, 224, 224), measure_latency=None, print_info=True):
    net_info = {}
    if isinstance(net, nn.DataParallel):
        net = net.module

    if "OFA" not in net.config["name"]:   # flops, calculations works only makes sense for static subnetworks
        net_info['flops'] = count_net_flops(net, [1] + list(input_shape)) / 1e6

    # parameters
    net_info['params'] = count_parameters(net) / 1e6

    # latencies
    latency_types = [] if measure_latency is None else measure_latency.split("#")
    for l_type in latency_types:
        latency, measured_latency = measure_net_latency(
            net, l_type, fast=False, input_shape=input_shape
        )
        net_info["%s latency" % l_type] = {"val": latency, "hist": measured_latency}

    if print_info:
        # print(net)
        print('Total training params: %.2fM' % (net_info['params']))
        if "flops" in net_info:
            print('Total FLOPs: %.2fM' % (net_info['flops']))
        for l_type in latency_types:
            print('Estimated %s latency: %.3fms' % (l_type, net_info["%s latency" % l_type]["val"]))

    return net_info



