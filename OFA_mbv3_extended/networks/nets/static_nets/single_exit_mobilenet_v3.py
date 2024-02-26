from torch import nn

from OFA_mbv3_extended.networks.nets.my_networks import MyNewNetwork
from OFA_mbv3_extended.networks.stages.static_stages.stat_stages import *
from OFA_mbv3_extended.networks.stages.static_stages.stat_stages_builders import *

__all__ = [
    "Single_Exit_MobileNetV3",
    "SE_B_MobileNetV3",
    "SE_D_MobileNetV3",
    "SE_P_MobileNetV3",
    "SE_DP_MobileNetV3",
    "SE_B_MobileNetV3_builder",
    "SE_D_MobileNetV3_builder",
    "SE_P_MobileNetV3_builder",
    "SE_DP_MobileNetV3_builder",
    "se_build_from_cfg"
]


class Single_Exit_MobileNetV3(MyNewNetwork):

    def __init__(
        self,
        head_stage,
        body_stages,
        exit_stage,
    ):

        super(Single_Exit_MobileNetV3, self).__init__()
        self.head_stage = head_stage
        self.body_stages = nn.ModuleList(body_stages)
        self.exit_stage = exit_stage

    def forward(self, x):
        x = self.head_stage(x)
        for body_stage in self.body_stages:
            x = body_stage(x)
        x = self.exit_stage(x)
        return x

    @property
    def name(self):
        return Single_Exit_MobileNetV3.__name__

    @property
    def module_str(self):
        _str = ""
        _str = self.head_stage.module_str + "\n"
        for body_stage in self.body_stages:
            _str += body_stage.module_str   # /n already present in stage
        _str += self.exit_stage.module_str
        return _str

    @property
    def config(self):
        return {
            "name": self.name,
            "bn": self.get_bn_param(),
            "head_stage": self.head_stage.config,
            "body_stages": [body_stage.config for body_stage in self.body_stages],
            "exit_stage": self.exit_stage.config,
        }

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

########################################################################################################################


def SE_B_MobileNetV3_builder(
    n_classes=200,
    bn_param=(0.1, 1e-5),
    dropout_rate=0.2,
    width_mult=1.0,
    ks=7,
    expand_ratio=6,
    depth_param=4
):
    # build head stage
    head_stage = static_head_stage_builder(width_mult)

    # build body stages
    body_stages = static_body_stages_builder(width_mult, ks, expand_ratio, depth_param)

    # build exit stage
    exit_stage = static_last_exit_stage_builder(width_mult, dropout_rate, n_classes)

    net = SE_B_MobileNetV3(head_stage, body_stages, exit_stage)

    net.set_bn_param(*bn_param)

    return net


class SE_B_MobileNetV3(Single_Exit_MobileNetV3):

    def __init__(
            self,
            head_stage,
            body_stages,
            exit_stage
    ):

        super(SE_B_MobileNetV3, self).__init__(head_stage, body_stages, exit_stage)

    @property
    def name(self):
        return SE_B_MobileNetV3.__name__

    @staticmethod
    def build_from_config(config):

        net = se_build_from_cfg(config, StaticBodyStage, SE_B_MobileNetV3)
        return net


########################################################################################################################


def SE_D_MobileNetV3_builder(
    n_classes=200,
    bn_param=(0.1, 1e-5),
    dropout_rate=0.2,
    width_mult=1.0,
    ks=7,
    expand_ratio=6,
    depth_param=4
):
    # build head stage
    head_stage = static_head_stage_builder(width_mult)

    # build body stages
    body_stages = static_dense_body_stages_builder(width_mult, ks, expand_ratio, depth_param)

    # build exit stage
    exit_stage = static_last_exit_stage_builder(width_mult, dropout_rate, n_classes)

    net = SE_D_MobileNetV3(head_stage, body_stages, exit_stage)

    # set bn param
    net.set_bn_param(*bn_param)

    return net


class SE_D_MobileNetV3(Single_Exit_MobileNetV3):

    def __init__(
            self,
            head_stage,
            body_stages,
            exit_stage
    ):

        super(SE_D_MobileNetV3, self).__init__(head_stage, body_stages, exit_stage)

    @property
    def name(self):
        return SE_D_MobileNetV3.__name__

    @staticmethod
    def build_from_config(config):

        net = se_build_from_cfg(config, StaticDenseBodyStage, SE_D_MobileNetV3)
        return net


########################################################################################################################


def SE_P_MobileNetV3_builder(
    n_classes=200,
    bn_param=(0.1, 1e-5),
    dropout_rate=0.2,
    width_mult=1.0,
    ks=7,
    expand_ratio=6,
    depth_param=4
):
    # build head stage
    head_stage = static_head_stage_builder(width_mult)

    # build body stages
    body_stages = static_parallel_body_stages_builder(width_mult, ks, expand_ratio, depth_param)

    # build exit stage
    exit_stage = static_last_exit_stage_builder(width_mult, dropout_rate, n_classes)

    net = SE_P_MobileNetV3(head_stage, body_stages, exit_stage)

    # set bn param
    net.set_bn_param(*bn_param)

    return net


class SE_P_MobileNetV3(Single_Exit_MobileNetV3):

    def __init__(
            self,
            head_stage,
            body_stages,
            exit_stage
    ):

        super(SE_P_MobileNetV3, self).__init__(head_stage, body_stages, exit_stage)

    @property
    def name(self):
        return SE_P_MobileNetV3.__name__

    @staticmethod
    def build_from_config(config):

        net = se_build_from_cfg(config, StaticParallelBodyStage, SE_P_MobileNetV3)
        return net


########################################################################################################################


def SE_DP_MobileNetV3_builder(
    n_classes=200,
    bn_param=(0.1, 1e-5),
    dropout_rate=0.2,
    width_mult=1.0,
    ks=7,
    expand_ratio=6,
    depth_param=4,
):

    # build head stage
    head_stage = static_head_stage_builder(width_mult)

    # build body stages
    body_stages = static_dense_parallel_body_stages_builder(width_mult, ks, expand_ratio, depth_param)

    # build exit stage
    exit_stage = static_last_exit_stage_builder(width_mult, dropout_rate, n_classes)

    net = SE_DP_MobileNetV3(head_stage, body_stages, exit_stage)

    # set bn param
    net.set_bn_param(*bn_param)

    return net


class SE_DP_MobileNetV3(Single_Exit_MobileNetV3):

    def __init__(
            self,
            head_stage,
            body_stages,
            exit_stage
    ):
        super(SE_DP_MobileNetV3, self).__init__(head_stage, body_stages, exit_stage)

    @property
    def name(self):
        return SE_DP_MobileNetV3.__name__

    @staticmethod
    def build_from_config(config):
        net = se_build_from_cfg(config, StaticDenseParallelBodyStage, SE_DP_MobileNetV3)
        return net
###############################################################


def se_build_from_cfg(config, stage_type, net_type):

    head_stage = StaticHeadStage.build_from_config(config["head_stage"])

    body_stages = []
    for body_stage_config in config["body_stages"]:
        body_stage = stage_type.build_from_config(body_stage_config)
        body_stages.append(body_stage)

    exit_stage = StaticExitStage.build_from_config(config["exit_stage"])

    net = net_type(head_stage, body_stages, exit_stage)

    if "bn" in config:
        net.set_bn_param(**config["bn"])
    else:
        net.set_bn_param(momentum=0.1, eps=1e-5)

    return net
