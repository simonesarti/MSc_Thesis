import copy
import random

from ofa.utils import val2list
from torch import nn

from OFA_mbv3_extended.networks.nets.dynamic_nets.utils import (
    check_and_reformat_ks_e_d_helper,
    check_expand_single_list
)
from OFA_mbv3_extended.networks.nets.my_networks import (
    MyDynamicNetwork,
    get_valid_nw_keys,
)
from OFA_mbv3_extended.networks.nets.static_nets.single_exit_mobilenet_v3 import *
from OFA_mbv3_extended.networks.stages.dynamic_stages.dyn_stages import *
from OFA_mbv3_extended.networks.stages.dynamic_stages.dyn_stages_builders import *
from OFA_mbv3_extended.networks.stages.static_stages.stat_stages_builders import (
    static_head_stage_builder, static_all_exit_stages_builders
)
from OFA_mbv3_extended.utils.common_tools import partition_list
from .utils import stages_settings_to_list, stages_settings_to_list_nw

__all__ = [
    "Early_Exit_OFAMobileNetV3",

    "EE_narrow_OFAMobileNetV3",
    "EE_B_OFAMobileNetV3",
    "EE_D_OFAMobileNetV3",

    "EE_wide_OFAMobileNetV3",
    "EE_P_OFAMobileNetV3",
    "EE_DP_OFAMobileNetV3",

    "ee_get_active_subnet",
    "ee_get_active_net_config"
]


class Early_Exit_OFAMobileNetV3(MyDynamicNetwork):

    def __init__(
            self,
            head_stage,
            body_stages,
            exit_stages,
            width_mult=1.0,
            ks_list=3,
            expand_ratio_list=6,
            depth_list=4,
            net_depth_list=5,
    ):

        super(Early_Exit_OFAMobileNetV3, self).__init__()
        self.head_stage = head_stage
        self.body_stages = nn.ModuleList(body_stages)
        self.exit_stages = nn.ModuleList(exit_stages)

        self.width_mult = width_mult
        self.ks_list = val2list(ks_list, 1)
        self.expand_ratio_list = val2list(expand_ratio_list, 1)
        self.depth_list = val2list(depth_list, 1)
        self.net_depth_list = val2list(net_depth_list, 1)

        self.ks_list.sort()
        self.expand_ratio_list.sort()
        self.depth_list.sort()
        self.net_depth_list.sort()

        self.active_exit = max(self.net_depth_list)

    def set_active_exit(self, active_exit):
        self.active_exit = active_exit

    def get_active_exit(self):
        return self.active_exit

    def get_active_exit_idx(self):
        return self.active_exit-1

    def forward(self, x):

        x = self.head_stage(x)
        for i in range(self.get_active_exit()):
            x = self.body_stages[i](x)
        x = self.exit_stages[self.get_active_exit_idx()](x)

        return x

    @property
    def module_str(self):

        _str = self.head_stage.module_str + "\n"
        for i in range(self.get_active_exit()):
            _str += self.body_stages[i].module_str
        _str += self.exit_stages[self.get_active_exit_idx()].module_str

        return _str

    @property
    def config(self):
        return {
            "name": self.name,
            "bn": self.get_bn_param(),
            "active_exit": self.active_exit,
            "head_stage": self.head_stage.config,
            "body_stages": [body_stage.config for body_stage in self.body_stages],
            "exit_stages": [exit_stage.config for exit_stage in self.exit_stages]
        }

    @property
    def name(self):
        return Early_Exit_OFAMobileNetV3.__name__

    def re_organize_middle_weights(self, expand_ratio_stage=0):

        for body_stage in self.body_stages:
            body_stage.re_organize_middle_weights(expand_ratio_stage)

    """non implemented methods"""

    # implemented in narrow/wide
    def set_max_net(self):
        raise NotImplementedError

    # implemented in narrow/wide
    def set_active_subnet(self, ks=None, e=None, d=None, **kwargs):
        raise NotImplementedError

    # implemented in narrow/wide
    def sample_active_subnet(self):
        raise NotImplementedError

    # implemented in narrow/wide
    def check_set_active_subnet_format(self, ks=None, e=None, d=None, **kwargs):
        raise NotImplementedError

    # implemented in specific net
    def get_active_subnet(self, preserve_weight=True):
        raise NotImplementedError

    # implemented in specific net
    def get_active_net_config(self):
        raise NotImplementedError

    # implemented in specific net
    def get_all_exits_subnet(self, preserve_weight=True):
        raise NotImplementedError

    # implemented in specific net
    def get_some_exits_subnet(self, exits_pos, preserve_weight=True):
        raise NotImplementedError

########################################################################################################################


class EE_narrow_OFAMobileNetV3(Early_Exit_OFAMobileNetV3):

    def __init__(
            self,
            head_stage,
            body_stages,
            exit_stages,
            width_mult=1.0,
            ks_list=3,
            expand_ratio_list=6,
            depth_list=4,
            net_depth_list=5,
    ):
        super(EE_narrow_OFAMobileNetV3, self).__init__(
            head_stage,
            body_stages,
            exit_stages,
            width_mult,
            ks_list,
            expand_ratio_list,
            depth_list,
            net_depth_list
        )

    @property
    def name(self):
        return EE_narrow_OFAMobileNetV3.__name__

    def sample_active_subnet(self):

        ks_candidates = self.ks_list
        expand_candidates = self.expand_ratio_list
        depth_candidates = self.depth_list

        stages_settings = []
        for body_stage in self.body_stages:
            stage_setting = body_stage.sample_active_substage(
                ks_candidates=ks_candidates,
                expand_candidates=expand_candidates,
                depth_candidates=depth_candidates
            )
            stages_settings.append(stage_setting)

        stages_settings_list = stages_settings_to_list(stages_settings)

        sampled_net_depth = random.choice(self.net_depth_list)
        self.set_active_exit(sampled_net_depth)

        stages_settings_list["nd"] = sampled_net_depth

        return stages_settings_list

    def set_max_net(self):

        self.set_active_subnet(
            ks=max(self.ks_list),
            e=max(self.expand_ratio_list),
            d=max(self.depth_list),
            nd=max(self.net_depth_list),
        )

    def set_active_subnet(self, ks=None, e=None, d=None, **kwargs):

        staged_ks, staged_e, staged_d = self.check_and_reformat(ks, e, d, **kwargs)

        for body_stage, stage_ks, stage_e, stage_d in zip(self.body_stages, staged_ks, staged_e, staged_d):
            body_stage.set_active_substage(ks=stage_ks, e=stage_e, d=stage_d)

        self.set_active_exit(kwargs["nd"])

    def check_and_reformat(self, ks=None, e=None, d=None, **kwargs):

        n_stages = len(self.body_stages)
        ks, e, d = check_and_reformat_ks_e_d_helper(
                                                    n_stages,
                                                    ks,
                                                    e,
                                                    d,
                                                    self.ks_list,
                                                    self.expand_ratio_list,
                                                    self.depth_list
        )

        if "nd" not in kwargs:
            raise ValueError('net depth (nd) field must be specified for EE networks')

        if not isinstance(kwargs["nd"], int):
            raise ValueError('nd field must be integer')

        if not kwargs["nd"] in self.net_depth_list:
            raise ValueError(f"net depth must in {self.net_depth_list}")

        return ks, e, d

    """ non implemented methods """

    def get_active_subnet(self, preserve_weight=True):
        raise NotImplementedError

    def get_active_net_config(self):
        raise NotImplementedError

    def get_all_exits_subnet(self, preserve_weight=True):
        raise NotImplementedError

    def get_some_exits_subnet(self, exits_pos, preserve_weight=True):
        raise NotImplementedError


class EE_B_OFAMobileNetV3(EE_narrow_OFAMobileNetV3):

    def __init__(
            self,
            n_classes=200,
            bn_param=(0.1, 1e-5),
            dropout_rate=0.1,
            width_mult=1.0,
            ks_list=7,
            expand_ratio_list=6,
            depth_list=4,
            net_depth_list=5,
    ):

        # build head stage
        head_stage = static_head_stage_builder(width_mult)

        # build body stages
        body_stages = dynamic_body_stages_builder(width_mult, ks_list, expand_ratio_list, max(depth_list))

        # build exit stage
        exit_stages = static_all_exit_stages_builders(width_mult, dropout_rate, n_classes)

        super(EE_B_OFAMobileNetV3, self).__init__(
            head_stage,
            body_stages,
            exit_stages,
            width_mult,
            ks_list,
            expand_ratio_list,
            depth_list,
            net_depth_list
        )

        # set bn param
        self.set_bn_param(*bn_param)

    @property
    def name(self):
        return EE_B_OFAMobileNetV3.__name__

    def get_active_subnet(self, preserve_weight=True):

        subnet = ee_get_active_subnet(self, SE_B_MobileNetV3, preserve_weight)
        return subnet

    def get_active_net_config(self):
        cfg = ee_get_active_net_config(self, SE_B_MobileNetV3)
        return cfg

    def get_all_exits_subnet(self, preserve_weight=True):
        from OFA_mbv3_extended.networks.nets.static_nets.early_exit_mobilenet_v3 import EE_B_MobileNetV3
        subnet = ee_get_all_exits_subnet(self, EE_B_MobileNetV3, preserve_weight)
        return subnet

    def get_some_exits_subnet(self, exits_pos, preserve_weight=True):
        from OFA_mbv3_extended.networks.nets.static_nets.single_multi_exit_mobilenet_v3 import SME_B_MobileNetV3
        subnet = ee_get_some_exits_subnet(self, SME_B_MobileNetV3, exits_pos, preserve_weight)
        return subnet


class EE_D_OFAMobileNetV3(EE_narrow_OFAMobileNetV3):

    def __init__(
            self,
            n_classes=200,
            bn_param=(0.1, 1e-5),
            dropout_rate=0.1,
            width_mult=1.0,
            ks_list=7,
            expand_ratio_list=6,
            depth_list=4,
            net_depth_list=5,
    ):

        # build head stage
        head_stage = static_head_stage_builder(width_mult)

        # build body stages
        body_stages = dynamic_dense_body_stages_builder(width_mult, ks_list, expand_ratio_list, max(depth_list))

        # build exit stage
        exit_stages = static_all_exit_stages_builders(width_mult, dropout_rate, n_classes)

        super(EE_D_OFAMobileNetV3, self).__init__(
            head_stage,
            body_stages,
            exit_stages,
            width_mult,
            ks_list,
            expand_ratio_list,
            depth_list,
            net_depth_list
        )

        # set bn param
        self.set_bn_param(*bn_param)

    @property
    def name(self):
        return EE_D_OFAMobileNetV3.__name__

    def get_active_subnet(self, preserve_weight=True):

        subnet = ee_get_active_subnet(self, SE_D_MobileNetV3, preserve_weight)
        return subnet

    def get_active_net_config(self):
        cfg = ee_get_active_net_config(self, SE_D_MobileNetV3)
        return cfg

    def get_all_exits_subnet(self, preserve_weight=True):
        from OFA_mbv3_extended.networks.nets.static_nets.early_exit_mobilenet_v3 import EE_D_MobileNetV3
        subnet = ee_get_all_exits_subnet(self, EE_D_MobileNetV3, preserve_weight)
        return subnet

    def get_some_exits_subnet(self, exits_pos, preserve_weight=True):
        from OFA_mbv3_extended.networks.nets.static_nets.single_multi_exit_mobilenet_v3 import SME_D_MobileNetV3
        subnet = ee_get_some_exits_subnet(self, SME_D_MobileNetV3, exits_pos, preserve_weight)
        return subnet

########################################################################################################################

class EE_wide_OFAMobileNetV3(Early_Exit_OFAMobileNetV3):

    def __init__(
            self,
            head_stage,
            body_stages,
            exit_stages,
            width_mult=1.0,
            ks_list=3,
            expand_ratio_list=6,
            depth_list=4,
            net_width_list=3,
            net_depth_list=5,

    ):

        super(EE_wide_OFAMobileNetV3, self).__init__(
            head_stage,
            body_stages,
            exit_stages,
            width_mult,
            ks_list,
            expand_ratio_list,
            depth_list,
            net_depth_list
        )

        self.net_width_list = val2list(net_width_list, 1)
        self.net_width_list.sort()

    @property
    def name(self):
        return EE_wide_OFAMobileNetV3.__name__

    def sample_active_subnet(self):

        ks_candidates = self.ks_list
        expand_candidates = self.expand_ratio_list
        depth_candidates = self.depth_list
        net_width_candidates = get_valid_nw_keys(self.net_width_list)

        stages_settings = []
        for body_stage in self.body_stages:
            stage_setting = body_stage.sample_active_substage(
                ks_candidates=ks_candidates,
                expand_candidates=expand_candidates,
                depth_candidates=depth_candidates,
                net_width_candidates=net_width_candidates    # set depending on k-v pairs in stage
            )
            stages_settings.append(stage_setting)

        stages_settings_list = stages_settings_to_list_nw(stages_settings)

        sampled_net_depth = random.choice(self.net_depth_list)
        self.set_active_exit(sampled_net_depth)

        stages_settings_list["nd"] = sampled_net_depth

        return stages_settings_list

    def set_max_net(self):

        self.set_active_subnet(
            ks=max(self.ks_list),
            e=max(self.expand_ratio_list),
            d=max(self.depth_list),
            nd=max(self.net_depth_list),
            nw=max(get_valid_nw_keys(max(self.net_width_list)))
        )

    def set_active_subnet(self, ks=None, e=None, d=None, **kwargs):

        staged_ks, staged_e, staged_d, staged_nw = self.check_and_reformat(ks, e, d, **kwargs)

        for body_stage, stage_ks, stage_e, stage_d, stage_nw \
                in zip(self.body_stages, staged_ks, staged_e, staged_d, staged_nw):
            body_stage.set_active_substage(ks=stage_ks, e=stage_e, d=stage_d, nw=stage_nw)

        self.set_active_exit(kwargs["nd"])

    def check_and_reformat(self, ks=None, e=None, d=None, **kwargs):

        n_stages = len(self.body_stages)
        ks, e, d = check_and_reformat_ks_e_d_helper(
            n_stages,
            ks,
            e,
            d,
            self.ks_list,
            self.expand_ratio_list,
            self.depth_list
        )

        # net depth
        if "nd" not in kwargs:
            raise ValueError('net depth (nd) field must be specified for EE networks')

        if not isinstance(kwargs["nd"], int):
            raise ValueError('nd field must be integer')

        if not kwargs["nd"] in self.net_depth_list:
            raise ValueError(f"net depth must in {self.net_depth_list}")

        # net width
        if "nw" not in kwargs:
            raise ValueError("net-width nd must be specified")

        stage_depth = max(self.depth_list)
        net_depth = n_stages * stage_depth

        allowed_nw = get_valid_nw_keys(self.net_width_list)

        nw = check_expand_single_list(kwargs["nw"], net_depth, allowed_nw)
        nw = partition_list(nw, n_stages)
        
        return ks, e, d, nw

    """ non implemented methods """

    def get_active_subnet(self, preserve_weight=True):
        raise NotImplementedError

    def get_active_net_config(self):
        raise NotImplementedError

    def get_all_exits_subnet(self, preserve_weight=True):
        raise NotImplementedError

    def get_some_exits_subnet(self, exits_pos, preserve_weight=True):
        raise NotImplementedError


class EE_P_OFAMobileNetV3(EE_wide_OFAMobileNetV3):

    def __init__(
            self,
            n_classes=200,
            bn_param=(0.1, 1e-5),
            dropout_rate=0.1,
            width_mult=1.0,
            ks_list=7,
            expand_ratio_list=6,
            depth_list=4,
            net_width_list=3,
            net_depth_list=5,
    ):

        # build head stage
        head_stage = static_head_stage_builder(width_mult)

        # build body stages
        body_stages = dynamic_parallel_body_stages_builder(width_mult, ks_list, expand_ratio_list, max(depth_list))

        # build exit stage
        exit_stages = static_all_exit_stages_builders(width_mult, dropout_rate, n_classes)

        super(EE_P_OFAMobileNetV3, self).__init__(
            head_stage,
            body_stages,
            exit_stages,
            width_mult,
            ks_list,
            expand_ratio_list,
            depth_list,
            net_width_list,
            net_depth_list,
        )

        # set bn param
        self.set_bn_param(*bn_param)

    @property
    def name(self):
        return EE_P_OFAMobileNetV3.__name__

    def get_active_subnet(self, preserve_weight=True):

        subnet = ee_get_active_subnet(self, SE_P_MobileNetV3, preserve_weight)
        return subnet

    def get_active_net_config(self):
        cfg = ee_get_active_net_config(self, SE_P_MobileNetV3)
        return cfg

    def get_all_exits_subnet(self, preserve_weight=True):
        from OFA_mbv3_extended.networks.nets.static_nets.early_exit_mobilenet_v3 import EE_P_MobileNetV3
        subnet = ee_get_all_exits_subnet(self, EE_P_MobileNetV3, preserve_weight)
        return subnet

    def get_some_exits_subnet(self, exits_pos, preserve_weight=True):
        from OFA_mbv3_extended.networks.nets.static_nets.single_multi_exit_mobilenet_v3 import SME_P_MobileNetV3
        subnet = ee_get_some_exits_subnet(self, SME_P_MobileNetV3, exits_pos, preserve_weight)
        return subnet


class EE_DP_OFAMobileNetV3(EE_wide_OFAMobileNetV3):

    def __init__(
            self,
            n_classes=200,
            bn_param=(0.1, 1e-5),
            dropout_rate=0.1,
            width_mult=1.0,
            ks_list=7,
            expand_ratio_list=6,
            depth_list=4,
            net_width_list=3,
            net_depth_list=5,
    ):

        # build head stage
        head_stage = static_head_stage_builder(width_mult)

        # build body stages
        body_stages = dynamic_dense_parallel_body_stages_builder(width_mult, ks_list, expand_ratio_list, max(depth_list))

        # build exit stage
        exit_stages = static_all_exit_stages_builders(width_mult, dropout_rate, n_classes)

        super(EE_DP_OFAMobileNetV3, self).__init__(
            head_stage,
            body_stages,
            exit_stages,
            width_mult,
            ks_list,
            expand_ratio_list,
            depth_list,
            net_width_list,
            net_depth_list
        )

        # set bn param
        self.set_bn_param(*bn_param)

    @property
    def name(self):
        return EE_DP_OFAMobileNetV3.__name__

    def get_active_subnet(self, preserve_weight=True):

        subnet = ee_get_active_subnet(self, SE_DP_MobileNetV3, preserve_weight)
        return subnet

    def get_active_net_config(self):
        cfg = ee_get_active_net_config(self, SE_DP_MobileNetV3)
        return cfg

    def get_all_exits_subnet(self, preserve_weight=True):
        from OFA_mbv3_extended.networks.nets.static_nets.early_exit_mobilenet_v3 import EE_DP_MobileNetV3
        subnet = ee_get_all_exits_subnet(self, EE_DP_MobileNetV3, preserve_weight)
        return subnet

    def get_some_exits_subnet(self, exits_pos, preserve_weight=True):
        from OFA_mbv3_extended.networks.nets.static_nets.single_multi_exit_mobilenet_v3 import SME_DP_MobileNetV3
        subnet = ee_get_some_exits_subnet(self, SME_DP_MobileNetV3, exits_pos, preserve_weight)
        return subnet


########################################################################################################################


def ee_get_active_subnet(dyn_net, static_net_type, preserve_weight=True):

    head_stage = copy.deepcopy(dyn_net.head_stage)

    active_body_stages = []
    prev_stage_out_ch = head_stage.first_layer.conv.out_channels

    for i in range(dyn_net.get_active_exit()):
        body_stage = dyn_net.body_stages[i]
        active_body_stages.append(body_stage.get_active_substage(prev_stage_out_ch, preserve_weight))
        prev_stage_out_ch = get_output_channels_from_dyn_stage(body_stage)

    active_exit_stage = copy.deepcopy(dyn_net.exit_stages[dyn_net.get_active_exit_idx()])

    subnet = static_net_type(head_stage, active_body_stages, active_exit_stage)
    subnet.set_bn_param(**dyn_net.get_bn_param())

    return subnet


def ee_get_active_net_config(dyn_net, static_net_type):

    body_stages_act_configs = []
    prev_stage_out_ch = dyn_net.head_stage.first_layer.conv.out_channels

    for i in range(dyn_net.get_active_exit()):
        body_stage = dyn_net.body_stages[i]
        act_config = body_stage.get_active_stage_config(prev_stage_out_ch)
        body_stages_act_configs.append(act_config)
        prev_stage_out_ch = get_output_channels_from_dyn_stage(body_stage)

    active_exit_stage_config = dyn_net.exit_stages[dyn_net.get_active_exit_idx()].config

    return {
        "name": static_net_type.__name__,
        "bn": dyn_net.get_bn_param(),
        "head_stage": dyn_net.head_stage.config,
        "body_stages": body_stages_act_configs,
        "exit_stage": active_exit_stage_config
    }


def ee_get_all_exits_subnet(dyn_net, static_net_type, preserve_weight=True):

    head_stage = copy.deepcopy(dyn_net.head_stage)

    body_stages = []
    prev_stage_out_ch = head_stage.first_layer.conv.out_channels
    for body_stage in dyn_net.body_stages:
        body_stages.append(body_stage.get_active_substage(prev_stage_out_ch, preserve_weight))
        prev_stage_out_ch = get_output_channels_from_dyn_stage(body_stage)

    exit_stages = []
    for exit_stage in dyn_net.exit_stages:
        exit_stages.append(copy.deepcopy(exit_stage))

    subnet = static_net_type(head_stage, body_stages, exit_stages)
    subnet.set_bn_param(**dyn_net.get_bn_param())

    return subnet


def ee_get_some_exits_subnet(dyn_net, static_net_type, exits_pos, preserve_weight=True):

    head_stage = copy.deepcopy(dyn_net.head_stage)

    body_stages = []
    prev_stage_out_ch = head_stage.first_layer.conv.out_channels
    for body_stage in dyn_net.body_stages:
        body_stages.append(body_stage.get_active_substage(prev_stage_out_ch, preserve_weight))
        prev_stage_out_ch = get_output_channels_from_dyn_stage(body_stage)

    exit_stages = []
    for exit_stage in dyn_net.exit_stages:
        exit_stages.append(copy.deepcopy(exit_stage))

    kept_exit_stages = []
    for i in exits_pos:
        kept_exit_stages.append(exit_stages[i-1])

    subnet = static_net_type(head_stage, body_stages, kept_exit_stages, exits_pos)
    subnet.set_bn_param(**dyn_net.get_bn_param())

    return subnet

