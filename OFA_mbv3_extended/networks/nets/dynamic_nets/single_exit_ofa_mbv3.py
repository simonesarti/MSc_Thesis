import copy

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
    static_head_stage_builder,
    static_last_exit_stage_builder
)
from OFA_mbv3_extended.utils.common_tools import partition_list
from .utils import stages_settings_to_list, stages_settings_to_list_nw

__all__ = [
    "Single_Exit_OFAMobileNetV3",

    "SE_narrow_OFAMobileNetV3",
    "SE_B_OFAMobileNetV3",
    "SE_D_OFAMobileNetV3",

    "SE_wide_OFAMobileNetV3",
    "SE_P_OFAMobileNetV3",
    "SE_DP_OFAMobileNetV3",

    "se_get_active_subnet",
    "se_get_active_net_config"
]


class Single_Exit_OFAMobileNetV3(MyDynamicNetwork):

    def __init__(
            self,
            head_stage,
            body_stages,
            exit_stage,
            width_mult=1.0,
            ks_list=3,
            expand_ratio_list=6,
            depth_list=4,
    ):
        super(Single_Exit_OFAMobileNetV3, self).__init__()
        self.head_stage = head_stage
        self.body_stages = nn.ModuleList(body_stages)
        self.exit_stage = exit_stage

        self.width_mult = width_mult
        self.ks_list = val2list(ks_list, 1)
        self.expand_ratio_list = val2list(expand_ratio_list, 1)
        self.depth_list = val2list(depth_list, 1)

        self.ks_list.sort()
        self.expand_ratio_list.sort()
        self.depth_list.sort()

    def forward(self, x):
        x = self.head_stage(x)
        for body_stage in self.body_stages:
            x = body_stage(x)
        x = self.exit_stage(x)
        return x

    @property
    def name(self):
        return Single_Exit_OFAMobileNetV3.__name__

    @property
    def module_str(self):

        _str = self.head_stage.module_str + "\n"
        for body_stage in self.body_stages:
            _str += body_stage.module_str  # /n already present in stage
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

    def re_organize_middle_weights(self, expand_ratio_stage=0):

        for body_stage in self.body_stages:
            body_stage.re_organize_middle_weights(expand_ratio_stage)

    """ non implemented methods """

    # implemented in narrow/wide
    def sample_active_subnet(self):
        raise NotImplementedError

    # implemented in narrow/wide
    def set_max_net(self):
        raise NotImplementedError

    # implemented in narrow/wide
    def set_active_subnet(self, ks=None, e=None, d=None, **kwargs):
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

########################################################################################################################


class SE_narrow_OFAMobileNetV3(Single_Exit_OFAMobileNetV3):

    def __init__(
            self,
            head_stage,
            body_stages,
            exit_stage,
            width_mult=1.0,
            ks_list=3,
            expand_ratio_list=6,
            depth_list=4,
    ):
        super(SE_narrow_OFAMobileNetV3, self).__init__(
            head_stage,
            body_stages,
            exit_stage,
            width_mult,
            ks_list,
            expand_ratio_list,
            depth_list,
        )

    @property
    def name(self):
        return SE_narrow_OFAMobileNetV3.__name__

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
        return stages_settings_list

    def set_max_net(self):
        self.set_active_subnet(
            ks=max(self.ks_list),
            e=max(self.expand_ratio_list),
            d=max(self.depth_list)
        )

    def set_active_subnet(self, ks=None, e=None, d=None, **kwargs):

        staged_ks, staged_e, staged_d = self.check_and_reformat(ks=ks, e=e, d=d)

        for body_stage, stage_ks, stage_e, stage_d in zip(self.body_stages, staged_ks, staged_e, staged_d):
            body_stage.set_active_substage(ks=stage_ks, e=stage_e, d=stage_d)

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

        return ks, e, d

    """ non implemented methods """

    def get_active_subnet(self, preserve_weight=True):
        raise NotImplementedError

    def get_active_net_config(self):
        raise NotImplementedError


class SE_B_OFAMobileNetV3(SE_narrow_OFAMobileNetV3):

    def __init__(
            self,
            n_classes=200,
            bn_param=(0.1, 1e-5),
            dropout_rate=0.1,
            width_mult=1.0,
            ks_list=7,
            expand_ratio_list=6,
            depth_list=4,
    ):

        # build head stage
        head_stage = static_head_stage_builder(width_mult)

        # build body stages
        body_stages = dynamic_body_stages_builder(width_mult, ks_list, expand_ratio_list, max(depth_list))

        # build exit stage
        exit_stage = static_last_exit_stage_builder(width_mult, dropout_rate, n_classes)

        super(SE_B_OFAMobileNetV3, self).__init__(
            head_stage,
            body_stages,
            exit_stage,
            width_mult,
            ks_list,
            expand_ratio_list,
            depth_list,
        )

        # set bn param
        self.set_bn_param(*bn_param)

    @property
    def name(self):
        return SE_B_OFAMobileNetV3.__name__

    def get_active_subnet(self, preserve_weight=True):

        subnet = se_get_active_subnet(self, SE_B_MobileNetV3, preserve_weight)
        return subnet

    def get_active_net_config(self):
        cfg = se_get_active_net_config(self, SE_B_MobileNetV3)
        return cfg


class SE_D_OFAMobileNetV3(SE_narrow_OFAMobileNetV3):

    def __init__(
            self,
            n_classes=200,
            bn_param=(0.1, 1e-5),
            dropout_rate=0.1,
            width_mult=1.0,
            ks_list=7,
            expand_ratio_list=6,
            depth_list=4,
    ):

        # build head stage
        head_stage = static_head_stage_builder(width_mult)

        # build body stages
        body_stages = dynamic_dense_body_stages_builder(width_mult, ks_list, expand_ratio_list, max(depth_list))

        # build exit stage
        exit_stage = static_last_exit_stage_builder(width_mult, dropout_rate, n_classes)

        super(SE_D_OFAMobileNetV3, self).__init__(
            head_stage,
            body_stages,
            exit_stage,
            width_mult,
            ks_list,
            expand_ratio_list,
            depth_list,
        )

        # set bn param
        self.set_bn_param(*bn_param)

    @property
    def name(self):
        return SE_D_OFAMobileNetV3.__name__

    def get_active_subnet(self, preserve_weight=True):
        subnet = se_get_active_subnet(self, SE_D_MobileNetV3, preserve_weight)
        return subnet

    def get_active_net_config(self):
        cfg = se_get_active_net_config(self, SE_D_MobileNetV3)
        return cfg


#######################################################################################################################


class SE_wide_OFAMobileNetV3(Single_Exit_OFAMobileNetV3):

    def __init__(
            self,
            head_stage,
            body_stages,
            exit_stage,
            width_mult=1.0,
            ks_list=3,
            expand_ratio_list=6,
            depth_list=4,
            net_width_list=3
    ):
        super(SE_wide_OFAMobileNetV3, self).__init__(
            head_stage,
            body_stages,
            exit_stage,
            width_mult,
            ks_list,
            expand_ratio_list,
            depth_list,
        )

        self.net_width_list = val2list(net_width_list, 1)
        self.net_width_list.sort()

    @property
    def name(self):
        return SE_wide_OFAMobileNetV3.__name__

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
                net_width_candidates=net_width_candidates
            )
            stages_settings.append(stage_setting)

        stages_settings_list = stages_settings_to_list_nw(stages_settings)

        return stages_settings_list

    def set_max_net(self):
        self.set_active_subnet(
            ks=max(self.ks_list),
            e=max(self.expand_ratio_list),
            d=max(self.depth_list),
            nw=max(get_valid_nw_keys(max(self.net_width_list)))
        )

    def set_active_subnet(self, ks=None, e=None, d=None, **kwargs):

        staged_ks, staged_e, staged_d, staged_nw = self.check_and_reformat(ks, e, d, **kwargs)

        for body_stage, stage_ks, stage_e, stage_d, stage_nw \
                in zip(self.body_stages, staged_ks, staged_e, staged_d, staged_nw):

            body_stage.set_active_substage(ks=stage_ks, e=stage_e, d=stage_d, nw=stage_nw)

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


class SE_P_OFAMobileNetV3(SE_wide_OFAMobileNetV3):

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
    ):

        # build head stage
        head_stage = static_head_stage_builder(width_mult)

        # build body stages (net 3 wide by design)
        body_stages = dynamic_parallel_body_stages_builder(width_mult, ks_list, expand_ratio_list, max(depth_list))

        # build exit stage
        exit_stage = static_last_exit_stage_builder(width_mult, dropout_rate, n_classes)

        super(SE_P_OFAMobileNetV3, self).__init__(
            head_stage,
            body_stages,
            exit_stage,
            width_mult,
            ks_list,
            expand_ratio_list,
            depth_list,
            net_width_list,
        )

        # set bn param
        self.set_bn_param(*bn_param)

    @property
    def name(self):
        return SE_P_OFAMobileNetV3.__name__

    def get_active_subnet(self, preserve_weight=True):
        subnet = se_get_active_subnet(self, SE_P_MobileNetV3, preserve_weight)
        return subnet

    def get_active_net_config(self):
        cfg = se_get_active_net_config(self, SE_P_MobileNetV3)
        return cfg


class SE_DP_OFAMobileNetV3(SE_wide_OFAMobileNetV3):

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
    ):

        # build head stage
        head_stage = static_head_stage_builder(width_mult)

        # build body stages
        body_stages = dynamic_dense_parallel_body_stages_builder(width_mult, ks_list, expand_ratio_list, max(depth_list))

        # build exit stage
        exit_stage = static_last_exit_stage_builder(width_mult, dropout_rate, n_classes)

        super(SE_DP_OFAMobileNetV3, self).__init__(
            head_stage,
            body_stages,
            exit_stage,
            width_mult,
            ks_list,
            expand_ratio_list,
            depth_list,
            net_width_list,
        )

        # set bn param
        self.set_bn_param(*bn_param)

    @property
    def name(self):
        return SE_DP_OFAMobileNetV3.__name__

    def get_active_subnet(self, preserve_weight=True):
        subnet = se_get_active_subnet(self, SE_DP_MobileNetV3, preserve_weight)
        return subnet

    def get_active_net_config(self):
        cfg = se_get_active_net_config(self, SE_DP_MobileNetV3)
        return cfg


#############################################################################################


def se_get_active_subnet(dyn_net, static_net_type, preserve_weight=True):

    head_stage = copy.deepcopy(dyn_net.head_stage)

    body_stages = []
    prev_stage_out_ch = head_stage.first_layer.conv.out_channels
    for body_stage in dyn_net.body_stages:
        body_stages.append(body_stage.get_active_substage(prev_stage_out_ch, preserve_weight))
        prev_stage_out_ch = get_output_channels_from_dyn_stage(body_stage)

    exit_stage = copy.deepcopy(dyn_net.exit_stage)

    subnet = static_net_type(head_stage, body_stages, exit_stage)
    subnet.set_bn_param(**dyn_net.get_bn_param())

    return subnet


def se_get_active_net_config(dyn_net, static_net_type):

    body_stages_act_configs = []
    prev_stage_out_ch = dyn_net.head_stage.first_layer.conv.out_channels
    for body_stage in dyn_net.body_stages:
        act_config = body_stage.get_active_stage_config(prev_stage_out_ch)
        body_stages_act_configs.append(act_config)
        prev_stage_out_ch = get_output_channels_from_dyn_stage(body_stage)

    return {
        "name": static_net_type.__name__,
        "bn": dyn_net.get_bn_param(),
        "head_stage": dyn_net.head_stage.config,
        "body_stages": body_stages_act_configs,
        "exit_stage": dyn_net.exit_stage.config
    }