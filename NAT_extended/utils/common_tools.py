import math

import numpy as np


# get weighted sum of values in the list
def list_weighted_sum(x, weights):
    if len(x) == 1:
        return x[0] * weights[0]
    else:
        return x[0] * weights[0] + list_weighted_sum(x[1:], weights[1:])


# get product of values in the list
def list_mul(x):
    if len(x) == 1:
        return x[0]
    else:
        return x[0] * list_mul(x[1:])


# if index is int, returns the value in value indexed from the list
# if index is fractional, returns  values in between the two that are not really indexed
def list_continuous_index(val_list, index):
    assert index <= len(val_list) - 1
    left_id = int(index)
    right_id = int(math.ceil(index))
    if left_id == right_id:
        return val_list[left_id]
    else:
        return val_list[left_id] * (right_id - index) + val_list[right_id] * (index - left_id)


# check if i==j
def delta_ij(i, j):
    if i == j:
        return 1
    else:
        return 0


# given an integer val, generates a list containing that integer repeat_time times
# if val is already a list,np.array,tuple, the list is returned directly
def int2list(val, repeat_time=1):
    if isinstance(val, list) or isinstance(val, np.ndarray):
        return val
    elif isinstance(val, tuple):
        return list(val)
    else:
        return [val for _ in range(repeat_time)]


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    # maxk = max(topk)
    maxk = min(max(topk), int(output.size()[1]))  # in case dataset has less than 5 classes
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        if k > maxk:
            k = maxk
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
