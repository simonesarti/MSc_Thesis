from torch import nn

from OFA_mbv3_extended.networks.nets.my_networks import MyNewNetwork
from OFA_mbv3_extended.networks.stages.static_stages.stat_stages import *
from OFA_mbv3_extended.networks.stages.static_stages.stat_stages_builders import *

__all__ = [
    "Early_Exit_MobileNetV3",
    "EE_B_MobileNetV3",
    "EE_D_MobileNetV3",
    "EE_P_MobileNetV3",
    "EE_DP_MobileNetV3",
    "EE_B_MobileNetV3_builder",
    "EE_D_MobileNetV3_builder",
    "EE_P_MobileNetV3_builder",
    "EE_DP_MobileNetV3_builder",
    "ee_build_from_cfg"
]


class Early_Exit_MobileNetV3(MyNewNetwork):

    def __init__(
            self,
            head_stage,
            body_stages,
            exit_stages,
    ):
        super(Early_Exit_MobileNetV3, self).__init__()
        self.head_stage = head_stage
        self.body_stages = nn.ModuleList(body_stages)
        self.exit_stages = nn.ModuleList(exit_stages)

    def forward(self, x):

        x = self.head_stage(x)
        y = []
        for body_stage in self.body_stages:
            x = body_stage(x)
            y.append(x)

        for i in range(len(self.body_stages)):
            y[i] = self.exit_stages[i](y[i])

        return y

    @property
    def name(self):
        return Early_Exit_MobileNetV3.__name__

    @property
    def module_str(self):
        _str = ""
        _str += self.head_stage.module_str + "\n"
        for i in range(len(self.body_stages)):
            _str += self.body_stages[i].module_str
            _str += self.exit_stages[i].module_str + "\n"
        return _str

    @property
    def config(self):
        return {
            "name": self.name,
            "bn": self.get_bn_param(),
            "head_stage": self.head_stage.config,
            "body_stages": [body_stage.config for body_stage in self.body_stages],
            "exit_stages": [exit_stage.config for exit_stage in self.exit_stages],
        }

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

########################################################################################################################


def EE_B_MobileNetV3_builder(
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

    # build exit stages
    exit_stages = static_all_exit_stages_builders(width_mult, dropout_rate, n_classes)

    net = EE_B_MobileNetV3(head_stage, body_stages, exit_stages)

    # set bn param
    net.set_bn_param(*bn_param)

    return net


class EE_B_MobileNetV3(Early_Exit_MobileNetV3):

    def __int__(
            self,
            head_stage,
            body_stages,
            exit_stages
    ):
        super(EE_B_MobileNetV3, self).__init__(head_stage, body_stages, exit_stages)

    @property
    def name(self):
        return EE_B_MobileNetV3.__name__

    @staticmethod
    def build_from_config(config):
        net = ee_build_from_cfg(config, StaticBodyStage, EE_B_MobileNetV3)
        return net


########################################################################################################################


def EE_D_MobileNetV3_builder(
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

    # build exit stages
    exit_stages = static_all_exit_stages_builders(width_mult, dropout_rate, n_classes)

    net = EE_D_MobileNetV3(head_stage, body_stages, exit_stages)

    # set bn param
    net.set_bn_param(*bn_param)

    return net


class EE_D_MobileNetV3(Early_Exit_MobileNetV3):

    def __int__(
            self,
            head_stage,
            body_stages,
            exit_stages
    ):
        super(EE_D_MobileNetV3, self).__init__(head_stage, body_stages, exit_stages)

    @property
    def name(self):
        return EE_D_MobileNetV3.__name__

    @staticmethod
    def build_from_config(config):
        net = ee_build_from_cfg(config, StaticDenseBodyStage, EE_D_MobileNetV3)
        return net


########################################################################################################################


def EE_P_MobileNetV3_builder(
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

    # build exit stages
    exit_stages = static_all_exit_stages_builders(width_mult, dropout_rate, n_classes)

    net = EE_P_MobileNetV3(head_stage, body_stages, exit_stages)

    # set bn param
    net.set_bn_param(*bn_param)

    return net


class EE_P_MobileNetV3(Early_Exit_MobileNetV3):

    def __int__(
            self,
            head_stage,
            body_stages,
            exit_stages
    ):
        super(EE_P_MobileNetV3, self).__init__(head_stage, body_stages, exit_stages)

    @property
    def name(self):
        return EE_P_MobileNetV3.__name__

    @staticmethod
    def build_from_config(config):
        net = ee_build_from_cfg(config, StaticParallelBodyStage, EE_P_MobileNetV3)
        return net


########################################################################################################################


def EE_DP_MobileNetV3_builder(
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
    body_stages = static_dense_parallel_body_stages_builder(width_mult, ks, expand_ratio, depth_param)

    # build exit stages
    exit_stages = static_all_exit_stages_builders(width_mult, dropout_rate, n_classes)

    net = EE_DP_MobileNetV3(head_stage, body_stages, exit_stages)

    # set bn param
    net.set_bn_param(*bn_param)

    return net


class EE_DP_MobileNetV3(Early_Exit_MobileNetV3):

    def __int__(
            self,
            head_stage,
            body_stages,
            exit_stages
    ):
        super(EE_DP_MobileNetV3, self).__init__(head_stage, body_stages, exit_stages)

    @property
    def name(self):
        return EE_DP_MobileNetV3.__name__

    @staticmethod
    def build_from_config(config):

        net = ee_build_from_cfg(config, StaticDenseParallelBodyStage, EE_DP_MobileNetV3)
        return net


#######################################################################################################################


def ee_build_from_cfg(config, stage_type, net_type):

    head_stage = StaticHeadStage.build_from_config(config["head_stage"])

    body_stages = []
    for body_stage_config in config["body_stages"]:
        body_stage = stage_type.build_from_config(body_stage_config)
        body_stages.append(body_stage)

    exit_stages = []
    for exit_stage_config in config["exit_stages"]:
        exit_stage = StaticExitStage.build_from_config(exit_stage_config)
        exit_stages.append(exit_stage)

    net = net_type(head_stage, body_stages, exit_stages)

    if "bn" in config:
        net.set_bn_param(**config["bn"])
    else:
        net.set_bn_param(momentum=0.1, eps=1e-5)

    return net
