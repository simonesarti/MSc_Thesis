from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import DynamicMBConvLayer
from ofa.utils import val2list
from ofa.utils.layers import ResidualBlock, IdentityLayer

from OFA_mbv3_extended.networks.stages.static_stages.stat_layers_builders import (
    conv1x1_layers_builder,
    transformation_layers_builder
)

__all__ = [
    "dynamic_body_stage_layers_builder",
    "dynamic_dense_body_stage_layers_builder",
    "dynamic_parallel_body_stage_layers_builder",
    "dynamic_dense_parallel_body_stage_layers_builder",
]


def dynamic_body_stage_layers_builder(
    depth,
    in_channels,
    out_channels,
    ks_list,
    expand_ratio_list,
    stride,
    act_func,
    use_se
):

    layers = []

    for i in range(depth):

        dyn_mb_conv = DynamicMBConvLayer(
            in_channel_list=val2list(in_channels) if i == 0 else val2list(out_channels),
            out_channel_list=val2list(out_channels),
            kernel_size_list=ks_list,
            expand_ratio_list=expand_ratio_list,
            stride=stride if i == 0 else 1,
            act_func=act_func,
            use_se=use_se,
        )

        if i != 0:
            shortcut = IdentityLayer(out_channels, out_channels)
        else:
            shortcut = None

        layers.append(ResidualBlock(dyn_mb_conv, shortcut))

    return layers


def dynamic_dense_body_stage_layers_builder(
    depth,
    in_channels,
    out_channels,
    ks_list,
    expand_ratio_list,
    stride,
    act_func,
    use_se
):

    return dynamic_body_stage_layers_builder(
        depth,
        in_channels,
        out_channels,
        ks_list,
        expand_ratio_list,
        stride,
        act_func,
        use_se
    )


def dynamic_parallel_body_stage_layers_builder(
    depth,
    in_channels,
    out_channels,
    ks_list,
    expand_ratio_list,
    stride,
    act_func,
    use_se
):

    # first layers are still the dynamicMBConv
    dyn_mb_conv_list = dynamic_body_stage_layers_builder(
        depth,
        in_channels,
        out_channels,
        ks_list,
        expand_ratio_list,
        stride,
        act_func,
        use_se
    )

    # added some static layers in parallel
    conv1x1_list = conv1x1_layers_builder(
        depth,
        in_channels,
        out_channels,
        stride,
        act_func
    )

    # attention_conv_list = attention_conv_layers_builder(
    #    depth,
    #    in_channels,
    #    out_channels,
    #    max(ks_list),
    #    stride
    # )

    transformation_layers_list = transformation_layers_builder(
        depth,
        in_channels,
        out_channels,
        stride,
        act_func
    )

    layers = []
    for i in range(depth):
        # layers.append([dyn_mb_conv_list[i], conv1x1_list[i], attention_conv_list[i]])
        layers.append([dyn_mb_conv_list[i], conv1x1_list[i], transformation_layers_list[i]])

    return layers


def dynamic_dense_parallel_body_stage_layers_builder(
    depth,
    in_channels,
    out_channels,
    ks_list,
    expand_ratio_list,
    stride,
    act_func,
    use_se
):

    return dynamic_parallel_body_stage_layers_builder(
        depth,
        in_channels,
        out_channels,
        ks_list,
        expand_ratio_list,
        stride,
        act_func,
        use_se
    )
