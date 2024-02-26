from OFA_mbv3_extended.networks.nets.my_networks import NetworksParameters
from OFA_mbv3_extended.networks.stages.static_stages.stat_layers_builders import *
from OFA_mbv3_extended.networks.stages.static_stages.stat_stages import *

__all__ = [
    "static_head_stage_builder",
    "static_body_stages_builder",
    "static_dense_body_stages_builder",
    "static_parallel_body_stages_builder",
    "static_dense_parallel_body_stages_builder",
    "static_last_exit_stage_builder",
    "static_all_exit_stages_builders",
]


def static_head_stage_builder(width_mult):

    values = NetworksParameters(width_mult).get_layers_values()

    first_conv, first_layer = static_head_stage_layers_builder(
        channels=values["head_stage_channels"],
        kernel_sizes=values["head_stage_ks"],
        expand_ratio=values["head_stage_expand_ratio"],
        strides=values["head_stage_strides"],
        act=values["head_stage_act"],
        use_se=values["head_stage_use_se"]
    )
    head_stage = StaticHeadStage(first_conv, first_layer)

    return head_stage


def static_body_stages_builder(width_mult, ks, expand_ratio, depth):

    values = NetworksParameters(width_mult).get_layers_values()

    body_stages = []
    for i in range(5):
        layers = static_body_stage_layers_builder(
            depth=depth,
            in_channels=values["head_stage_channels"][-1] if i == 0 else values["body_stages_channels"][i - 1],
            out_channels=values["body_stages_channels"][i],
            kernel_size=ks,
            expand_ratio=expand_ratio,
            stride=values["body_stages_strides"][i],
            act=values["body_stages_act"][i],
            use_se=values["body_stages_use_se"][i]
        )
        body_stage = StaticBodyStage(layers)
        body_stages.append(body_stage)

    return body_stages


def static_dense_body_stages_builder(width_mult, ks, expand_ratio, depth):

    values = NetworksParameters(width_mult).get_layers_values()

    body_stages = []
    for i in range(5):
        layers = static_dense_body_stage_layers_builder(
            depth=depth,
            in_channels=values["head_stage_channels"][-1] if i == 0 else values["body_stages_channels"][i - 1],
            out_channels=values["body_stages_channels"][i],
            kernel_size=ks,
            expand_ratio=expand_ratio,
            stride=values["body_stages_strides"][i],
            act=values["body_stages_act"][i],
            use_se=values["body_stages_use_se"][i]
        )
        body_stage = StaticDenseBodyStage(layers)
        body_stages.append(body_stage)

    return body_stages


def static_parallel_body_stages_builder(width_mult, ks, expand_ratio, depth):

    values = NetworksParameters(width_mult).get_layers_values()

    body_stages = []
    for i in range(5):
        layer_blocks = static_parallel_body_stage_layers_builder(
            depth=depth,
            in_channels=values["head_stage_channels"][-1] if i == 0 else values["body_stages_channels"][i - 1],
            out_channels=values["body_stages_channels"][i],
            kernel_size=ks,
            expand_ratio=expand_ratio,
            stride=values["body_stages_strides"][i],
            act=values["body_stages_act"][i],
            use_se=values["body_stages_use_se"][i]
        )

        body_stage = StaticParallelBodyStage(layer_blocks)
        body_stages.append(body_stage)

    return body_stages


def static_dense_parallel_body_stages_builder(width_mult, ks, expand_ratio, depth):

    values = NetworksParameters(width_mult).get_layers_values()

    body_stages = []
    for i in range(5):
        layer_blocks = static_dense_parallel_body_stage_layers_builder(
            depth=depth,
            in_channels=values["head_stage_channels"][-1] if i == 0 else values["body_stages_channels"][i - 1],
            out_channels=values["body_stages_channels"][i],
            kernel_size=ks,
            expand_ratio=expand_ratio,
            stride=values["body_stages_strides"][i],
            act=values["body_stages_act"][i],
            use_se=values["body_stages_use_se"][i]
        )

        body_stage = StaticDenseParallelBodyStage(layer_blocks)
        body_stages.append(body_stage)

    return body_stages


def static_last_exit_stage_builder(width_mult, dropout_rate, n_classes):

    values = NetworksParameters(width_mult).get_layers_values()

    final_expand_layer, feature_mix_layer, classifier = static_exit_stage_layers_builder(
        in_channels=values["body_stages_channels"][-1],
        out_channels=values["exit5_stage_channels"],
        kernel_sizes=values["exit_stage_ks"],
        act=values["exit_stage_act"],
        dropout_rate=dropout_rate,
        n_classes=n_classes
    )
    exit_stage = StaticExitStage(final_expand_layer, feature_mix_layer, classifier)

    return exit_stage


def static_all_exit_stages_builders(width_mult, dropout_rate, n_classes):

    values = NetworksParameters(width_mult).get_layers_values()

    exit_stages = []
    for i in range(5):
        final_expand_layer, feature_mix_layer, classifier = static_exit_stage_layers_builder(
            in_channels=values["body_stages_channels"][i],
            out_channels=values["exit%d_stage_channels" % (i + 1)],
            kernel_sizes=values["exit_stage_ks"],
            act=values["exit_stage_act"],
            dropout_rate=dropout_rate,
            n_classes=n_classes
        )
        exit_stage = StaticExitStage(final_expand_layer, feature_mix_layer, classifier)
        exit_stages.append(exit_stage)

    return exit_stages


