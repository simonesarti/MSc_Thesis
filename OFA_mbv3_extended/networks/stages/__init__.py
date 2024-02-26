from OFA_mbv3_extended.networks.stages.dynamic_stages.dyn_stages import *
from OFA_mbv3_extended.networks.stages.static_stages.stat_stages import *


def get_stage_from_name(name):

    if name == StaticHeadStage.__name__:
        return StaticHeadStage
    elif name == StaticBodyStage.__name__:
        return StaticBodyStage
    elif name == StaticExitStage.__name__:
        return StaticExitStage
    elif name == StaticDenseBodyStage.__name__:
        return StaticDenseBodyStage
    elif name == StaticParallelBodyStage.__name__:
        return StaticParallelBodyStage
    elif name == StaticDenseParallelBodyStage.__name__:
        return StaticDenseParallelBodyStage

    elif name == DynamicBodyStage.__name__:
        return DynamicBodyStage
    elif name == DynamicDenseBodyStage.__name__:
        return DynamicDenseBodyStage
    elif name == DynamicParallelBodyStage.__name__:
        return DynamicParallelBodyStage
    elif name == DynamicDenseParallelBodyStage.__name__:
        return DynamicDenseParallelBodyStage

    else:
        raise NotImplementedError
