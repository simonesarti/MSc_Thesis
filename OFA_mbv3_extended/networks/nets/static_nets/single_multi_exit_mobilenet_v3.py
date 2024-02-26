from torch import nn
from OFA_mbv3_extended.networks.nets.my_networks import MyNewNetwork
from OFA_mbv3_extended.networks.stages.static_stages.stat_stages import *
from OFA_mbv3_extended.networks.stages.static_stages.stat_stages_builders import *


__all__ = [
    "SME_MobileNetV3",
    "sme_build_from_cfg",
    #####################
    "SME_B_MobileNetV3",
    "SME_D_MobileNetV3",
    "SME_P_MobileNetV3",
    "SME_DP_MobileNetV3",
    #####################
    "SME_B_MobileNetV3_builder",
    "SME_D_MobileNetV3_builder",
    "SME_P_MobileNetV3_builder",
    "SME_DP_MobileNetV3_builder",
]

class SME_MobileNetV3(MyNewNetwork):

    def __init__(
            self,
            head_stage,
            body_stages: list,
            exit_stages: list,
            exits_positions: list
    ):
        super(SME_MobileNetV3, self).__init__()
        exits_positions = sorted(exits_positions)
        self.exits_positions = exits_positions
        self.exits_idxs = [exit_pos - 1 for exit_pos in exits_positions]

        self.head_stage = head_stage
        self.body_stages = nn.ModuleList(body_stages[:self.exits_positions[-1]])
        self.exit_stages = nn.ModuleList(exit_stages)

        self.is_single_exit = (len(exit_stages) == 1)

        self.safety_checks()

    def safety_checks(self):

        if len(self.exit_stages) != len(self.exits_positions):
            raise AttributeError(f"the number of exits must correspond to the number of positions,\
             got {len(self.exit_stages)} and {len(self.exits_positions)}")

        if len(self.exit_stages) > len(self.body_stages):
            raise AttributeError("the number of exits must be lesser or equal to that of stages")

        if len(set(self.exits_positions)) != len(self.exits_positions):
            raise AttributeError("positions cannot be repeated")

        for ep in self.exits_positions:
            if ep < 1:
                raise AttributeError("exits positions must be positive values")
            if ep > len(self.body_stages):
                raise AttributeError("exits positions cannot be higher than number of stages")

    def forward(self, x):

        x = self.head_stage(x)

        y = []
        for i in range(len(self.body_stages)):
            x = self.body_stages[i](x)
            if i in self.exits_idxs:
                exit_stage_idx = self.exits_idxs.index(i)
                y.append(self.exit_stages[exit_stage_idx](x))

        return y

    @property
    def name(self):
        return SME_MobileNetV3.__name__

    @property
    def module_str(self):
        _str = ""
        _str += self.head_stage.module_str + "\n"
        for i in range(len(self.body_stages)):
            _str += self.body_stages[i].module_str
            if i in self.exits_idxs:
                exit_stage_idx = self.exits_idxs.index(i)
                _str += self.exit_stages[exit_stage_idx].module_str + "\n"
        return _str

    @property
    def config(self):
        return {
            "name": self.name,
            "bn": self.get_bn_param(),
            "head_stage": self.head_stage.config,
            "body_stages": [body_stage.config for body_stage in self.body_stages],
            "exit_stages": [exit_stage.config for exit_stage in self.exit_stages],
            "exits_positions": self.exits_positions
        }

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError


def sme_build_from_cfg(config, stage_type, net_type):

    head_stage = StaticHeadStage.build_from_config(config["head_stage"])

    body_stages = []
    for body_stage_config in config["body_stages"]:
        body_stage = stage_type.build_from_config(body_stage_config)
        body_stages.append(body_stage)

    exit_stages = []
    for exit_stage_config in config["exit_stages"]:
        exit_stage = StaticExitStage.build_from_config(exit_stage_config)
        exit_stages.append(exit_stage)

    exits_positions = config["exits_positions"]

    net = net_type(head_stage, body_stages, exit_stages, exits_positions)

    if "bn" in config:
        net.set_bn_param(**config["bn"])
    else:
        net.set_bn_param(momentum=0.1, eps=1e-5)

    return net


########################################################################################################################
# TRUE NETWORKS
########################################################################################################################

class SME_B_MobileNetV3(SME_MobileNetV3):
    def __init__(
            self,
            head_stage,
            body_stages: list,
            exit_stages: list,
            exits_positions: list
    ):
        super(SME_B_MobileNetV3, self).__init__(head_stage, body_stages, exit_stages, exits_positions)

    @property
    def name(self):
        return SME_B_MobileNetV3.__name__

    @staticmethod
    def build_from_config(config):
        net = sme_build_from_cfg(config, StaticBodyStage, SME_B_MobileNetV3)
        return net


class SME_D_MobileNetV3(SME_MobileNetV3):
    def __init__(
            self,
            head_stage,
            body_stages: list,
            exit_stages: list,
            exits_positions: list
    ):
        super(SME_D_MobileNetV3, self).__init__(head_stage, body_stages, exit_stages, exits_positions)

    @property
    def name(self):
        return SME_D_MobileNetV3.__name__

    @staticmethod
    def build_from_config(config):
        net = sme_build_from_cfg(config, StaticDenseBodyStage, SME_D_MobileNetV3)
        return net


class SME_P_MobileNetV3(SME_MobileNetV3):
    def __init__(
            self,
            head_stage,
            body_stages: list,
            exit_stages: list,
            exits_positions: list
    ):
        super(SME_P_MobileNetV3, self).__init__(head_stage, body_stages, exit_stages, exits_positions)

    @property
    def name(self):
        return SME_P_MobileNetV3.__name__

    @staticmethod
    def build_from_config(config):
        net = sme_build_from_cfg(config, StaticParallelBodyStage, SME_P_MobileNetV3)
        return net


class SME_DP_MobileNetV3(SME_MobileNetV3):
    def __init__(
            self,
            head_stage,
            body_stages: list,
            exit_stages: list,
            exits_positions: list
    ):
        super(SME_DP_MobileNetV3, self).__init__(head_stage, body_stages, exit_stages, exits_positions)

    @property
    def name(self):
        return SME_DP_MobileNetV3.__name__

    @staticmethod
    def build_from_config(config):
        net = sme_build_from_cfg(config, StaticDenseParallelBodyStage, SME_DP_MobileNetV3)
        return net


########################################################################################################################
# NETWORKS BUILDERS
########################################################################################################################


def SME_B_MobileNetV3_builder(
        exits_positions,
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
    body_stages = static_body_stages_builder(width_mult, ks, expand_ratio, depth_param)

    # build exit stages
    exit_stages = static_all_exit_stages_builders(width_mult, dropout_rate, n_classes)

    net = SME_B_MobileNetV3(head_stage, body_stages, exit_stages, exits_positions)

    # set bn param
    net.set_bn_param(*bn_param)

    return net


def SME_D_MobileNetV3_builder(
    exits_positions,
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

    net = SME_D_MobileNetV3(head_stage, body_stages, exit_stages, exits_positions)

    # set bn param
    net.set_bn_param(*bn_param)

    return net


def SME_P_MobileNetV3_builder(
    exits_positions,
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

    net = SME_P_MobileNetV3(head_stage, body_stages, exit_stages, exits_positions)

    # set bn param
    net.set_bn_param(*bn_param)

    return net


def SME_DP_MobileNetV3_builder(
    exits_positions,
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

    net = SME_DP_MobileNetV3(head_stage, body_stages, exit_stages, exits_positions)

    # set bn param
    net.set_bn_param(*bn_param)

    return net


