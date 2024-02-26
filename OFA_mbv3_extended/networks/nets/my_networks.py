from ofa.utils import make_divisible
from ofa.utils.layers import ResidualBlock, MBConvLayer, IdentityLayer
from ofa.utils.my_modules import MyModule, set_bn_param, get_bn_param

__all__ = [
    "MyNewNetwork",
    "MyDynamicNetwork",
    "NetworksParameters",
    "get_teacher_by_name",
    "get_net_by_name",
    "initialize_dyn_net",
    "get_valid_nw_keys",
    "get_nw_active_state",
    "get_nw_encoding"

]


class MyNewNetwork(MyModule):

    CHANNEL_DIVISIBLE = 8

    def forward(self, x):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    """ implemented methods """

    def zero_last_gamma(self):
        for m in self.modules():
            if isinstance(m, ResidualBlock):
                if isinstance(m.conv, MBConvLayer) and isinstance(
                        m.shortcut, IdentityLayer
                ):
                    m.conv.point_linear.bn.weight.data.zero_()

    def set_bn_param(self, momentum, eps, gn_channel_per_group=None, **kwargs):
        set_bn_param(self, momentum, eps, gn_channel_per_group, **kwargs)

    def get_bn_param(self):
        return get_bn_param(self)

    def get_parameters(self, keys=None, mode="include"):
        if keys is None:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    yield param
        elif mode == "include":
            for name, param in self.named_parameters():
                flag = False
                for key in keys:
                    if key in name:
                        flag = True
                        break
                if flag and param.requires_grad:
                    yield param
        elif mode == "exclude":
            for name, param in self.named_parameters():
                flag = True
                for key in keys:
                    if key in name:
                        flag = False
                        break
                if flag and param.requires_grad:
                    yield param
        else:
            raise ValueError("do not support: %s" % mode)

    def weight_parameters(self):
        return self.get_parameters()


class MyDynamicNetwork(MyNewNetwork):

    def forward(self, x):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise ValueError("do not support this function")

    def set_max_net(self):
        raise NotImplementedError

    def set_active_subnet(self, ks=None, e=None, d=None, **kwargs):
        raise NotImplementedError

    def check_set_active_subnet_format(self,ks=None, e=None, d=None, **kwargs):
        raise NotImplementedError

    def get_active_subnet(self, preserve_weight=True):
        raise NotImplementedError

    def sample_active_subnet(self):
        raise NotImplementedError

    def get_active_net_config(self):
        raise NotImplementedError

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        raise NotImplementedError

    def load_state_dict(self, state_dict, **kwargs):
        model_dict = self.state_dict()

        for key in state_dict:
            if key not in model_dict:
                if ".conv.weight" in key:
                    new_key = key.replace("conv.weight", "conv.conv.weight")
                elif "bn." in key:
                    new_key = key.replace("bn.", "bn.bn.")
                else:
                    raise ValueError(f"{key} not in model state dict nor has a conversion step")
            else:
                new_key = key

            assert new_key in model_dict, "%s" % new_key
            model_dict[new_key] = state_dict[key]

        super(MyDynamicNetwork, self).load_state_dict(model_dict)


class NetworksParameters:

    def __init__(self, width_mult):

        self.width_mult = width_mult

        self.layers_values = {}

        self.layers_values["head_stage_channels"] = self.to_divisible([16, 16])
        self.layers_values["body_stages_channels"] = self.to_divisible([24, 40, 80, 112, 160])

        exit1_size = [
            self.layers_values["body_stages_channels"][0] * 6,
            self.layers_values["body_stages_channels"][0] * 8,
        ]  # [144, 192]

        exit2_size = [
            self.layers_values["body_stages_channels"][1] * 6,
            self.layers_values["body_stages_channels"][1] * 8,
        ]  # [240, 320]

        exit3_size = [
            self.layers_values["body_stages_channels"][2] * 6,
            self.layers_values["body_stages_channels"][2] * 8,
        ]  # [480, 640]

        exit4_size = [
            self.layers_values["body_stages_channels"][3] * 6,
            self.layers_values["body_stages_channels"][3] * 8,
        ]  # [672, 896]

        exit5_size = [
            self.layers_values["body_stages_channels"][4] * 6,
            self.layers_values["body_stages_channels"][4] * 8,
        ]  # [960, 1280]

        self.layers_values["exit1_stage_channels"] = self.to_divisible(exit1_size)
        self.layers_values["exit2_stage_channels"] = self.to_divisible(exit2_size)
        self.layers_values["exit3_stage_channels"] = self.to_divisible(exit3_size)
        self.layers_values["exit4_stage_channels"] = self.to_divisible(exit4_size)
        self.layers_values["exit5_stage_channels"] = self.to_divisible(exit5_size)

        self.layers_values["head_stage_ks"] = [3, 3]
        # self.values["body_stages_ks"]
        self.layers_values["exit_stage_ks"] = [1, 1]

        self.layers_values["head_stage_expand_ratio"] = 1
        # self.layers_values["body_stages_expand_ratio"]

        self.layers_values["head_stage_strides"] = [2, 1]
        self.layers_values["body_stages_strides"] = [2, 2, 2, 1, 2]
        self.layers_values["exit_stage_strides"] = [1, 1]

        self.layers_values["head_stage_act"] = ["h_swish", "relu"]
        self.layers_values["body_stages_act"] = ["relu", "relu", "h_swish", "h_swish", "h_swish"]
        self.layers_values["exit_stage_act"] = ["h_swish", "h_swish"]

        self.layers_values["head_stage_use_se"] = [False, False]
        self.layers_values["body_stages_use_se"] = [False, True, False, True, True]
        self.layers_values["exit_stage_use_se"] = [False, False]

    def to_divisible(self, width):

        if isinstance(width, int):
            return make_divisible(width * self.width_mult, MyNewNetwork.CHANNEL_DIVISIBLE)
        else:
            assert isinstance(width, list)
            divisible_width = []
            for w in width:
                new_w = make_divisible(w * self.width_mult, MyNewNetwork.CHANNEL_DIVISIBLE)
                divisible_width.append(new_w)
            return divisible_width

    def get_layers_values(self):
        return self.layers_values


def get_teacher_by_name(name):

    if name == "OFAMobileNetV3":
        from ofa.imagenet_classification.networks import MobileNetV3Large
        return MobileNetV3Large

    elif name == "SE_B_OFAMobileNetV3":
        from .static_nets.single_exit_mobilenet_v3 import SE_B_MobileNetV3_builder
        return SE_B_MobileNetV3_builder
    elif name == "SE_D_OFAMobileNetV3":
        from .static_nets.single_exit_mobilenet_v3 import SE_D_MobileNetV3_builder
        return SE_D_MobileNetV3_builder
    elif name == "SE_P_OFAMobileNetV3":
        from .static_nets.single_exit_mobilenet_v3 import SE_P_MobileNetV3_builder
        return SE_P_MobileNetV3_builder
    elif name == "SE_DP_OFAMobileNetV3":
        from .static_nets.single_exit_mobilenet_v3 import SE_DP_MobileNetV3_builder
        return SE_DP_MobileNetV3_builder

    # early exit dynamic_layers networks
    elif name == "EE_B_OFAMobileNetV3":
        from .static_nets.early_exit_mobilenet_v3 import EE_B_MobileNetV3_builder
        return EE_B_MobileNetV3_builder
    elif name == "EE_D_OFAMobileNetV3":
        from .static_nets.early_exit_mobilenet_v3 import EE_D_MobileNetV3_builder
        return EE_D_MobileNetV3_builder
    elif name == "EE_P_OFAMobileNetV3":
        from .static_nets.early_exit_mobilenet_v3 import EE_P_MobileNetV3_builder
        return EE_P_MobileNetV3_builder
    elif name == "EE_DP_OFAMobileNetV3":
        from .static_nets.early_exit_mobilenet_v3 import EE_DP_MobileNetV3_builder
        return EE_DP_MobileNetV3_builder


def get_net_by_name(name):

    # original net
    if name == "OFAMobileNetV3":
        from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3
        return OFAMobileNetV3
    elif name == "MobileNetV3Large":
        from ofa.imagenet_classification.networks import MobileNetV3Large
        return MobileNetV3Large

    # single exit static_layers networks
    elif name == "SE_B_MobileNetV3":
        from .static_nets.single_exit_mobilenet_v3 import SE_B_MobileNetV3
        return SE_B_MobileNetV3
    elif name == "SE_D_MobileNetV3":
        from .static_nets.single_exit_mobilenet_v3 import SE_D_MobileNetV3
        return SE_D_MobileNetV3
    elif name == "SE_P_MobileNetV3":
        from .static_nets.single_exit_mobilenet_v3 import SE_P_MobileNetV3
        return SE_P_MobileNetV3
    elif name == "SE_DP_MobileNetV3":
        from .static_nets.single_exit_mobilenet_v3 import SE_DP_MobileNetV3
        return SE_DP_MobileNetV3

    # early exit static_layers networks
    elif name == "EE_B_MobileNetV3":
        from .static_nets.early_exit_mobilenet_v3 import EE_B_MobileNetV3
        return EE_B_MobileNetV3
    elif name == "EE_D_MobileNetV3":
        from .static_nets.early_exit_mobilenet_v3 import EE_D_MobileNetV3
        return EE_D_MobileNetV3
    elif name == "EE_P_MobileNetV3":
        from .static_nets.early_exit_mobilenet_v3 import EE_P_MobileNetV3
        return EE_P_MobileNetV3
    elif name == "EE_DP_MobileNetV3":
        from .static_nets.early_exit_mobilenet_v3 import EE_DP_MobileNetV3
        return EE_DP_MobileNetV3

    # single exit dynamic_layers networks
    elif name == "SE_B_OFAMobileNetV3":
        from .dynamic_nets.single_exit_ofa_mbv3 import SE_B_OFAMobileNetV3
        return SE_B_OFAMobileNetV3
    elif name == "SE_D_OFAMobileNetV3":
        from .dynamic_nets.single_exit_ofa_mbv3 import SE_D_OFAMobileNetV3
        return SE_D_OFAMobileNetV3
    elif name == "SE_P_OFAMobileNetV3":
        from .dynamic_nets.single_exit_ofa_mbv3 import SE_P_OFAMobileNetV3
        return SE_P_OFAMobileNetV3
    elif name == "SE_DP_OFAMobileNetV3":
        from .dynamic_nets.single_exit_ofa_mbv3 import SE_DP_OFAMobileNetV3
        return SE_DP_OFAMobileNetV3

    # early exit dynamic_layers networks
    elif name == "EE_B_OFAMobileNetV3":
        from .dynamic_nets.early_exit_ofa_mbv3 import EE_B_OFAMobileNetV3
        return EE_B_OFAMobileNetV3
    elif name == "EE_D_OFAMobileNetV3":
        from .dynamic_nets.early_exit_ofa_mbv3 import EE_D_OFAMobileNetV3
        return EE_D_OFAMobileNetV3
    elif name == "EE_P_OFAMobileNetV3":
        from .dynamic_nets.early_exit_ofa_mbv3 import EE_P_OFAMobileNetV3
        return EE_P_OFAMobileNetV3
    elif name == "EE_DP_OFAMobileNetV3":
        from .dynamic_nets.early_exit_ofa_mbv3 import EE_DP_OFAMobileNetV3
        return EE_DP_OFAMobileNetV3

    else:
        raise ValueError("unrecognized type of network: %s" % name)


def initialize_dyn_net(n_classes, args):

    # values to initialize the network types
    n_classes = n_classes
    bn_param = (args.bn_momentum, args.bn_eps)
    dropout_rate = args.dropout
    width_mult = args.width_mult
    ks_list = args.ks_list
    expand_ratio_list = args.expand_list
    depth_list = args.depth_list
    net_depth_list = args.net_depth_list
    net_width_list = args.net_width_list

    # get network
    net_type = get_net_by_name(args.network)

    if args.network in ["OFAMobileNetV3", "SE_B_OFAMobileNetV3", "SE_D_OFAMobileNetV3"]:
        net = net_type(
            n_classes=n_classes,
            bn_param=bn_param,
            dropout_rate=dropout_rate,
            width_mult=width_mult,
            ks_list=ks_list,
            expand_ratio_list=expand_ratio_list,
            depth_list=depth_list
        )

    elif args.network in ["SE_P_OFAMobileNetV3", "SE_DP_OFAMobileNetV3"]:
        net = net_type(
            n_classes=n_classes,
            bn_param=bn_param,
            dropout_rate=dropout_rate,
            width_mult=width_mult,
            ks_list=ks_list,
            expand_ratio_list=expand_ratio_list,
            depth_list=depth_list,
            net_width_list=net_width_list
        )

    elif args.network in ["EE_B_OFAMobileNetV3", "EE_D_OFAMobileNetV3"]:
        net = net_type(
            n_classes=n_classes,
            bn_param=bn_param,
            dropout_rate=dropout_rate,
            width_mult=width_mult,
            ks_list=ks_list,
            expand_ratio_list=expand_ratio_list,
            depth_list=depth_list,
            net_depth_list=net_depth_list
        )

    elif args.network in ["EE_P_OFAMobileNetV3", "EE_DP_OFAMobileNetV3"]:
        net = net_type(
            n_classes=n_classes,
            bn_param=bn_param,
            dropout_rate=dropout_rate,
            width_mult=width_mult,
            ks_list=ks_list,
            expand_ratio_list=expand_ratio_list,
            depth_list=depth_list,
            net_depth_list=net_depth_list,
            net_width_list=net_width_list
        )

    else:
        raise ValueError("Network not available")

    return net


""" net width related methods """

nw_num_to_act_dict = {
    1: (False, False, True),    # 001
    2: (False, True, False),    # 010
    3: (False, True, True),    # 011
    4: (True, False, False),   # 100
    5: (True, False, True),   # 101
    6: (True, True, False),   # 110
    7: (True, True, True),   # 111
}

nw_act_to_num_dict = {
    (False, False, True): 1,  # 001
    (False, True, False): 2,  # 010
    (False, True, True): 3,     # 011
    (True, False, False): 4,   # 100
    (True, False, True): 5,   # 101
    (True, True, False): 6,   # 110
    (True, True, True): 7    # 111
}


def get_valid_nw_keys_single(nw):

    valid_keys = []
    for k, v in nw_num_to_act_dict.items():
        if sum(v) == nw:
            valid_keys.append(k)
    return valid_keys


def get_valid_nw_keys_multiple(nw_list):
    valid_keys = []
    for nw in nw_list:
        for k, v in nw_num_to_act_dict.items():
            if nw == sum(v):
                valid_keys.append(k)

    return valid_keys


def get_valid_nw_keys(nw_list):

    if isinstance(nw_list, int):
        valid_keys = get_valid_nw_keys_single(nw_list)
    elif isinstance(nw_list, list) and len(nw_list) == 1:
        valid_keys = get_valid_nw_keys_single(nw_list[0])
    elif isinstance(nw_list, list) and len(nw_list) > 1:
        valid_keys = get_valid_nw_keys_multiple(nw_list)
    else:
        raise ValueError

    return valid_keys


def get_nw_active_state(key):
    return nw_num_to_act_dict[key]


def get_nw_encoding(state):
    return nw_act_to_num_dict[state]

