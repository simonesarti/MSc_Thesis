from OFA_mbv3_extended.networks.nets.my_networks import NetworksParameters
from .dyn_layers_builders import *
from .dyn_stages import *

__all__ = [
    "dynamic_body_stages_builder",
    "dynamic_dense_body_stages_builder",
    "dynamic_parallel_body_stages_builder",
    "dynamic_dense_parallel_body_stages_builder",
]


def dynamic_body_stages_builder(width_mult, ks_list, expand_ratio_list, depth):

    values = NetworksParameters(width_mult).get_layers_values()

    body_stages = []
    for i in range(5):
        layers = dynamic_body_stage_layers_builder(
            depth=depth,
            in_channels=values["head_stage_channels"][-1] if i == 0 else values["body_stages_channels"][i - 1],
            out_channels=values["body_stages_channels"][i],
            ks_list=ks_list,
            expand_ratio_list=expand_ratio_list,
            stride=values["body_stages_strides"][i],
            act_func=values["body_stages_act"][i],
            use_se=values["body_stages_use_se"][i]
        )

        body_stage = DynamicBodyStage(layers)
        body_stages.append(body_stage)

    return body_stages


def dynamic_dense_body_stages_builder(width_mult, ks_list, expand_ratio_list, depth):

    values = NetworksParameters(width_mult).get_layers_values()

    body_stages = []
    for i in range(5):
        layers = dynamic_dense_body_stage_layers_builder(
            depth=depth,
            in_channels=values["head_stage_channels"][-1] if i == 0 else values["body_stages_channels"][i - 1],
            out_channels=values["body_stages_channels"][i],
            ks_list=ks_list,
            expand_ratio_list=expand_ratio_list,
            stride=values["body_stages_strides"][i],
            act_func=values["body_stages_act"][i],
            use_se=values["body_stages_use_se"][i]
        )

        body_stage = DynamicDenseBodyStage(layers)
        body_stages.append(body_stage)

    return body_stages


def dynamic_parallel_body_stages_builder(width_mult, ks_list, expand_ratio_list, depth):

    values = NetworksParameters(width_mult).get_layers_values()

    body_stages = []
    for i in range(5):
        layer_blocks = dynamic_parallel_body_stage_layers_builder(
            depth=depth,
            in_channels=values["head_stage_channels"][-1] if i == 0 else values["body_stages_channels"][i - 1],
            out_channels=values["body_stages_channels"][i],
            ks_list=ks_list,
            expand_ratio_list=expand_ratio_list,
            stride=values["body_stages_strides"][i],
            act_func=values["body_stages_act"][i],
            use_se=values["body_stages_use_se"][i]
        )

        body_stage = DynamicParallelBodyStage(layer_blocks)
        body_stages.append(body_stage)

    return body_stages


def dynamic_dense_parallel_body_stages_builder(width_mult, ks_list, expand_ratio_list, depth):

    values = NetworksParameters(width_mult).get_layers_values()

    body_stages = []
    for i in range(5):
        layer_blocks = dynamic_dense_parallel_body_stage_layers_builder(
            depth=depth,
            in_channels=values["head_stage_channels"][-1] if i == 0 else values["body_stages_channels"][i - 1],
            out_channels=values["body_stages_channels"][i],
            ks_list=ks_list,
            expand_ratio_list=expand_ratio_list,
            stride=values["body_stages_strides"][i],
            act_func=values["body_stages_act"][i],
            use_se=values["body_stages_use_se"][i]
        )

        body_stage = DynamicDenseParallelBodyStage(layer_blocks)
        body_stages.append(body_stage)

    return body_stages
