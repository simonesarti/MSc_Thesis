import copy
import random

from ofa.utils import val2list
from ofa.utils.layers import ResidualBlock, ConvLayer
from torch import nn

from OFA_mbv3_extended.networks.layers.static_layers.stat_layers import AttentionConv, TransformationLayer
from OFA_mbv3_extended.networks.nets.my_networks import get_nw_active_state
from OFA_mbv3_extended.networks.stages.static_stages.stat_stages import *
from OFA_mbv3_extended.utils.common_tools import get_input_tensor

__all__ = [
    "MyDynamicStage",
    "DynamicBodyStage",
    "DynamicDenseBodyStage",
    "DynamicParallelBodyStage",
    "DynamicDenseParallelBodyStage",
    "get_output_channels_from_dyn_stage"
]


class MyDynamicStage(MyStage):

    def forward(self, x):
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

    def set_max_stage(self, max_ks, max_e, max_d, **kwargs):
        raise NotImplementedError

    def set_active_substage(self, ks=None, e=None, d=None, **kwargs):
        raise NotImplementedError

    def get_active_substage(self, prev_stage_out_ch, preserve_weight=True):
        raise NotImplementedError

    def sample_active_substage(self, ks_candidates, expand_candidates, depth_candidates, **kwargs):
        raise NotImplementedError

    def get_active_stage_config(self, prev_stage_out_ch):
        raise NotImplementedError

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        raise NotImplementedError


class DynamicBodyStage(MyDynamicStage):

    def __init__(
            self,
            layers,
    ):
        super(DynamicBodyStage, self).__init__()

        self.layers = nn.ModuleList(layers)
        self.n_active = len(self.layers)

    def get_n_active(self):
        return self.n_active

    def set_n_active(self, n_active):
        self.n_active = n_active

    def forward(self, x):

        for i in range(self.n_active):
            x = self.layers[i](x)
        return x

    @property
    def module_str(self):
        _str = ""
        for i in range(self.n_active):
            _str += self.layers[i].module_str + "\n"
        return _str

    @property
    def config(self):
        return {
            "name": DynamicBodyStage.__name__,
            "n_active": self.get_n_active(),
            "layers": [layer.config for layer in self.layers]
        }

    def set_max_stage(self, max_ks, max_e, max_d, **kwargs):
        self.set_active_substage(max_ks, max_e, max_d)

    def set_active_substage(self, ks=None, e=None, d=None, **kwargs):

        ks = val2list(ks, len(self.layers))
        expand_ratio = val2list(e, len(self.layers))

        for layer, ks, e in zip(self.layers, ks, expand_ratio):
            if ks is not None:
                layer.conv.active_kernel_size = ks
            if e is not None:
                layer.conv.active_expand_ratio = e

        self.set_n_active(d)

    def get_active_substage(self, prev_stage_out_ch, preserve_weight=True):

        new_layers = []
        input_channels = prev_stage_out_ch
        for i in range(self.n_active):

            new_layers.append(
                ResidualBlock(
                    self.layers[i].conv.get_active_subnet(input_channels, preserve_weight),
                    copy.deepcopy(self.layers[i].shortcut),
                )
            )
            input_channels = new_layers[-1].conv.out_channels

        substage = StaticBodyStage(new_layers)
        return substage

    def sample_active_substage(self, ks_candidates, expand_candidates, depth_candidates, **kwargs):

        # sample kernel size
        ks_setting = []
        ks_candidates = [ks_candidates for _ in range(len(self.layers))]
        for k_set in ks_candidates:
            k = random.choice(k_set)
            ks_setting.append(k)

        # sample expand ratio
        expand_setting = []
        expand_candidates = [expand_candidates for _ in range(len(self.layers))]
        for e_set in expand_candidates:
            e = random.choice(e_set)
            expand_setting.append(e)

        # sample depth
        depth_setting = random.choice(depth_candidates)

        self.set_active_substage(ks=ks_setting, e=expand_setting, d=depth_setting)

        return {
            "ks": ks_setting,
            "e": expand_setting,
            "d": depth_setting,
        }

    def get_active_stage_config(self, prev_stage_out_ch):

        new_layers_config_list = []
        input_channels = prev_stage_out_ch
        for i in range(self.n_active):
            new_layers_config_list.append(
                {
                    "name": ResidualBlock.__name__,
                    "conv": self.layers[i].conv.get_active_subnet_config(input_channels),
                    "shortcut": self.layers[i].shortcut.config if self.layers[i].shortcut is not None else None,
                }
            )
            input_channels = self.layers[i].conv.active_out_channel

        return {
            "name": StaticBodyStage.__name__,
            "layers": new_layers_config_list
        }

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        for layer in self.layers:
            layer.conv.re_organize_middle_weights(expand_ratio_stage)


class DynamicDenseBodyStage(DynamicBodyStage):

    def __init__(
            self,
            layers,
    ):
        super(DynamicDenseBodyStage, self).__init__(layers)

    def forward(self, x):

        depth = self.n_active
        y = [x]
        for d in range(depth):
            input_tensor = get_input_tensor(d, y)
            y.append(self.layers[d](input_tensor))

        return get_input_tensor(depth, y)

    # @property
    # def module_str(self):
    # ==> inherited from DynamicBodyStage

    @property
    def config(self):
        return {
            "name": DynamicDenseBodyStage.__name__,
            "n_active": self.get_n_active(),
            "layers": [layer.config for layer in self.layers]
        }

    # def def set_max_stage(self, max_ks, max_e, max_d, **kwargs):
    # ==> inherited from DynamicBodyStage

    # def set_active_substage(self, ks=None, e=None, d=None, **kwargs):
    # ==> inherited from DynamicBodyStage

    def get_active_substage(self, prev_stage_out_ch, preserve_weight=True):

        new_layers = []
        input_channels = prev_stage_out_ch
        for i in range(self.n_active):
            new_layers.append(
                ResidualBlock(
                    self.layers[i].conv.get_active_subnet(input_channels, preserve_weight),
                    copy.deepcopy(self.layers[i].shortcut),
                )
            )
            input_channels = new_layers[-1].conv.out_channels

        substage = StaticDenseBodyStage(new_layers)
        return substage

    # def sample_active_substage(self, ks_candidates, expand_candidates, depth_candidates, **kwargs):
    # ==> inherited from DynamicBodyStage

    def get_active_stage_config(self, prev_stage_out_ch):

        new_layers_config_list = []
        input_channels = prev_stage_out_ch
        for i in range(self.n_active):
            new_layers_config_list.append(
                {
                    "name": ResidualBlock.__name__,
                    "conv": self.layers[i].conv.get_active_subnet_config(input_channels),
                    "shortcut": self.layers[i].shortcut.config if self.layers[i].shortcut is not None else None,
                }
            )
            input_channels = self.layers[i].conv.active_out_channel

        return {
            "name": StaticDenseBodyStage.__name__,
            "layers": new_layers_config_list
        }

    # def re_organize_middle_weights(self, expand_ratio_stage=0):
    # ==> inherited from DynamicBodyStage


class DynamicParallelBodyStage(MyDynamicStage):

    def __init__(
            self,
            layers_blocks,
    ):

        super(DynamicParallelBodyStage, self).__init__()

        self.layers_blocks = nn.ModuleList()
        for group_of_parallel_layers in layers_blocks:
            self.layers_blocks.append(nn.ModuleList(group_of_parallel_layers))

        self.n_active_blocks = len(self.layers_blocks)

        self.layers_activation_status = []
        for group_of_parallel_layers in self.layers_blocks:
            self.layers_activation_status.append([True]*len(group_of_parallel_layers))

    def get_n_active_blocks(self):
        return self.n_active_blocks

    def set_n_active_blocks(self, n_active_blocks):
        self.n_active_blocks = n_active_blocks

    def get_layers_activation_status(self):
        return self.layers_activation_status

    # set for specific group of parallel layers
    def set_activation_layer_block(self, layers_block_idx, status):
        self.layers_activation_status[layers_block_idx] = status

    # set all groups at the same time
    def set_layers_activations(self, status):
        self.layers_activation_status = status

    def forward(self, x):

        active_layers_blocks = self.layers_blocks[:self.n_active_blocks]
        for block_idx, active_layer_block in enumerate(active_layers_blocks):

            parallel_results = []

            # pass input tensor x through all the parallel layers,
            # save the parallel results
            for layer_idx_in_group, layer in enumerate(active_layer_block):
                if self.layers_activation_status[block_idx][layer_idx_in_group] is True:
                    parallel_results.append(layer(x))

            # add the obtained tensors back into x, fed to next group
            x = 0
            for res in parallel_results:
                x += res

        return x

    @property
    def module_str(self):

        active_layers_blocks = self.layers_blocks[:self.n_active_blocks]

        _str = ""
        for block_idx, active_layers_block in enumerate(active_layers_blocks):
            for layer_idx_in_group, layer in enumerate(active_layers_block):
                if self.layers_activation_status[block_idx][layer_idx_in_group] is True:
                    _str += layer.module_str + " -- "
            _str += "\n"

        return _str

    @property
    def config(self):

        layers_blocks_configs = []
        for layers_block in self.layers_blocks:
            layers_configs = [layer.config for layer in layers_block]
            layers_blocks_configs.append(layers_configs)

        return {
            "name": DynamicParallelBodyStage.__name__,
            "n_active_blocks": self.n_active_blocks,
            "layers_activation_status": self.layers_activation_status,
            "layers_blocks": layers_blocks_configs

        }

    def set_max_stage(self, max_ks, max_e, max_d, **kwargs):
        # kwargs should contain max_nw
        max_nw = kwargs["max_nw"]
        self.set_active_substage(ks=max_ks, e=max_e, d=max_d, nw=max_nw)

    def set_active_substage(self, ks=None, e=None, d=None, **kwargs):
        nw = kwargs["nw"]

        ks = val2list(ks, len(self.layers_blocks))
        expand_ratio = val2list(e, len(self.layers_blocks))
        net_width = val2list(nw, len(self.layers_blocks))

        for group_idx, (group_of_parallel_layers, ks, e, nw) in enumerate(zip(self.layers_blocks, ks, expand_ratio, net_width)):

            # ks and expand influence the dynamic conv layers, not the other 2 if they remain static 1x1 and attention/transformation
            dynamic_conv_layer = group_of_parallel_layers[0]
            if ks is not None:
                dynamic_conv_layer.conv.active_kernel_size = ks
            if e is not None:
                dynamic_conv_layer.conv.active_expand_ratio = e

            # activate only some layers
            status_list = list(get_nw_active_state(nw))
            self.set_activation_layer_block(group_idx, status_list)

        self.set_n_active_blocks(d)

    def get_active_substage(self, prev_stage_out_ch, preserve_weight=True):
        new_layers_blocks = []
        input_channels = prev_stage_out_ch

        active_layers_blocks = self.layers_blocks[:self.n_active_blocks]
        for group_idx, group_of_parallel_layers in enumerate(active_layers_blocks):
            new_group = []
            for layer_idx, layer in enumerate(group_of_parallel_layers):
                if self.layers_activation_status[group_idx][layer_idx] is True:

                    if isinstance(layer, ResidualBlock):
                        new_group.append(
                            ResidualBlock(
                                self.layers_blocks[group_idx][layer_idx].conv.get_active_subnet(input_channels, preserve_weight),
                                copy.deepcopy(self.layers_blocks[group_idx][layer_idx].shortcut),
                            )
                        )
                    elif isinstance(layer, (ConvLayer, AttentionConv, TransformationLayer)):
                        new_group.append(
                            copy.deepcopy(layer)
                        )
                    else:
                        raise ValueError("layer type not found")

            input_channels = self.layers_blocks[group_idx][0].conv.out_channels
            new_layers_blocks.append(new_group)

        substage = StaticParallelBodyStage(new_layers_blocks)
        return substage

    def sample_active_substage(self, ks_candidates, expand_candidates, depth_candidates, **kwargs):

        # sample kernel size
        ks_setting = []
        ks_candidates = [ks_candidates for _ in range(len(self.layers_blocks))]
        for k_set in ks_candidates:
            k = random.choice(k_set)
            ks_setting.append(k)

        # sample expand ratio
        expand_setting = []
        expand_candidates = [expand_candidates for _ in range(len(self.layers_blocks))]
        for e_set in expand_candidates:
            e = random.choice(e_set)
            expand_setting.append(e)

        # sample net-width
        nw_candidates = kwargs["net_width_candidates"]
        net_width_setting = []
        net_width_candidates = [nw_candidates for _ in range(len(self.layers_blocks))]
        for nw_set in net_width_candidates:
            nw = random.choice(nw_set)
            net_width_setting.append(nw)

        # sample depth
        depth_setting = random.choice(depth_candidates)

        self.set_active_substage(ks=ks_setting, e=expand_setting, d=depth_setting, nw=net_width_setting)

        return {
            "ks": ks_setting,
            "e": expand_setting,
            "d": depth_setting,
            "nw": net_width_setting,
        }

    def get_active_stage_config(self, prev_stage_out_ch):
        new_layers_blocks_config_list = []
        input_channels = prev_stage_out_ch

        active_layers_blocks = self.layers_blocks[:self.n_active_blocks]

        for group_idx, group_of_parallel_layers in enumerate(active_layers_blocks):
            new_group_config_list = []

            for layer_idx, layer in enumerate(group_of_parallel_layers):
                if self.get_layers_activation_status()[group_idx][layer_idx] is True:

                    if isinstance(layer, ResidualBlock):
                        new_group_config_list.append(
                            {
                                "name": ResidualBlock.__name__,
                                "conv": self.layers_blocks[group_idx][layer_idx].conv.get_active_subnet_config(input_channels),
                                "shortcut": self.layers_blocks[group_idx][layer_idx].shortcut.config
                                if self.layers_blocks[group_idx][layer_idx].shortcut is not None
                                else None,
                            }
                        )
                    elif isinstance(layer, (ConvLayer, AttentionConv, TransformationLayer)):
                        new_group_config_list.append(
                            self.layers_blocks[group_idx][layer_idx].config
                        )
                    else:
                        raise ValueError("layer type not found")

            input_channels = self.layers_blocks[group_idx][0].conv.active_out_channel
            new_layers_blocks_config_list.append(new_group_config_list)

        return {
            "name": StaticParallelBodyStage.__name__,
            "layers_blocks": new_layers_blocks_config_list
        }

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        # only affect DynamicMBConvlayers
        for group_of_parallel_layers in self.layers_blocks:
            dynamic_conv = group_of_parallel_layers[0]
            dynamic_conv.conv.re_organize_middle_weights(expand_ratio_stage)


class DynamicDenseParallelBodyStage(DynamicParallelBodyStage):

    def __init__(
            self,
            layers_blocks,
    ):
        super(DynamicDenseParallelBodyStage, self).__init__(layers_blocks)

    def forward(self, x):

        depth = self.n_active_blocks
        y = [x]
        for d in range(depth):
            input_tensor = get_input_tensor(d, y)

            parallel_results = []
            # pass input tensor through active parallel layers, save parallel results
            for layer_idx, layer in enumerate(self.layers_blocks[d]):
                if self.layers_activation_status[d][layer_idx] is True:
                    parallel_results.append(layer(input_tensor))

            # add the obtained tensors
            x = 0
            for res in parallel_results:
                x += res

            y.append(x)

        return get_input_tensor(depth, y)

    # @property
    # def module_str(self):
    # ==> inherited from DynamicParallelBodyStage

    @property
    def config(self):

        layers_blocks_configs = []
        for layers_block in self.layers_blocks:
            layers_configs = [layer.config for layer in layers_block]
            layers_blocks_configs.append(layers_configs)

        return {
            "name": DynamicDenseParallelBodyStage.__name__,
            "n_active_blocks": self.n_active_blocks,
            "layers_activation_status": self.layers_activation_status,
            "layers_blocks": layers_blocks_configs
        }

    # def set_max_stage(self, max_ks, max_e, max_d, **kwargs):
    # ==> inherited from DynamicParallelBodyStage

    # def set_active_substage(self, ks=None, e=None, d=None, **kwargs):
    # ==> inherited from DynamicParallelBodyStage

    def get_active_substage(self, prev_stage_out_ch, preserve_weight=True):
        new_layers_blocks = []
        input_channels = prev_stage_out_ch

        active_layers_blocks = self.layers_blocks[:self.n_active_blocks]
        for group_idx, group_of_parallel_layers in enumerate(active_layers_blocks):
            new_group = []
            for layer_idx, layer in enumerate(group_of_parallel_layers):
                if self.layers_activation_status[group_idx][layer_idx] is True:

                    if isinstance(layer, ResidualBlock):
                        new_group.append(
                            ResidualBlock(
                                self.layers_blocks[group_idx][layer_idx].conv.get_active_subnet(input_channels, preserve_weight),
                                copy.deepcopy(self.layers_blocks[group_idx][layer_idx].shortcut),
                            )
                        )
                    elif isinstance(layer, (ConvLayer, AttentionConv, TransformationLayer)):
                        new_group.append(
                            copy.deepcopy(layer)
                        )
                    else:
                        raise ValueError("layer type not found")

            input_channels = self.layers_blocks[group_idx][0].conv.out_channels
            new_layers_blocks.append(new_group)

        substage = StaticDenseParallelBodyStage(new_layers_blocks)
        return substage

    # def sample_active_substage(self, ks_candidates, expand_candidates, depth_candidates, **kwargs):
    # ==> inherited from DynamicParallelBodyStage

    def get_active_stage_config(self, prev_stage_out_ch):
        new_layers_blocks_config_list = []
        input_channels = prev_stage_out_ch

        active_layers_blocks = self.layers_blocks[:self.n_active_blocks]

        for group_idx, group_of_parallel_layers in enumerate(active_layers_blocks):
            new_group_config_list = []

            for layer_idx, layer in enumerate(group_of_parallel_layers):
                if self.get_layers_activation_status()[group_idx][layer_idx] is True:

                    if isinstance(layer, ResidualBlock):
                        new_group_config_list.append(
                            {
                                "name": ResidualBlock.__name__,
                                "conv": self.layers_blocks[group_idx][layer_idx].conv.get_active_subnet_config(input_channels),
                                "shortcut": self.layers_blocks[group_idx][layer_idx].shortcut.config
                                if self.layers_blocks[group_idx][layer_idx].shortcut is not None
                                else None,
                            }
                        )
                    elif isinstance(layer, (ConvLayer, AttentionConv, TransformationLayer)):
                        new_group_config_list.append(
                            self.layers_blocks[group_idx][layer_idx].config
                        )
                    else:
                        raise ValueError("layer type not found")

            input_channels = self.layers_blocks[group_idx][0].conv.active_out_channel
            new_layers_blocks_config_list.append(new_group_config_list)

        return {
            "name": StaticDenseParallelBodyStage.__name__,
            "layers_blocks": new_layers_blocks_config_list
        }

    # def re_organize_middle_weights(self, expand_ratio_stage=0):
    # ==> inherited from DynamicParallelBodyStage


def get_output_channels_from_dyn_stage(stage):

    if isinstance(stage, DynamicBodyStage) or isinstance(stage, DynamicDenseBodyStage):
        out_ch = stage.layers[-1].conv.active_out_channel
    elif isinstance(stage, DynamicParallelBodyStage) or isinstance(stage, DynamicDenseParallelBodyStage):
        out_ch = stage.layers_blocks[-1][0].conv.active_out_channel
    else:
        raise NotImplementedError

    return out_ch





