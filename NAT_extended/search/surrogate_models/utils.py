import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

_DEBUG = False


def reset_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


def get_correlation(prediction, target):
    import scipy.stats as stats

    rmse = np.sqrt(((prediction - target) ** 2).mean())

    try:
        r, _ = stats.pearsonr(prediction[:, 0], target[:, 0])
    except IndexError:
        try:
            r, _ = stats.pearsonr(prediction, target)
        except ValueError:  # for rbf
            r, _ = stats.pearsonr(prediction[:, 0], target)

    rho, _ = stats.spearmanr(prediction, target)
    tau, _ = stats.kendalltau(prediction, target)

    return rmse, r, rho, tau

