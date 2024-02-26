from ofa.utils.layers import ConvLayer, MBConvLayer, ResidualBlock, IdentityLayer, LinearLayer

from OFA_mbv3_extended.networks.layers.static_layers.stat_layers import AttentionConv, TransformationLayer

__all__ = [
    "static_head_stage_layers_builder",
    "static_body_stage_layers_builder",
    "static_exit_stage_layers_builder",
    "static_dense_body_stage_layers_builder",
    "static_parallel_body_stage_layers_builder",
    "static_dense_parallel_body_stage_layers_builder",

    "conv1x1_layers_builder",
    "attention_conv_layers_builder",
    "transformation_layers_builder"
]


def static_head_stage_layers_builder(
    channels,
    kernel_sizes,
    expand_ratio,
    strides,
    act,
    use_se
):

    first_conv = ConvLayer(
        in_channels=3,
        out_channels=channels[0],
        kernel_size=kernel_sizes[0],
        stride=strides[0],
        act_func=act[0],
        use_se=use_se[0],
        use_bn=True,
        ops_order="weight_bn_act"
    )

    first_layer = MBConvLayer(
        in_channels=channels[0],
        out_channels=channels[1],
        kernel_size=kernel_sizes[1],
        stride=strides[1],
        expand_ratio=expand_ratio,
        act_func=act[1],
        use_se=use_se[1],
    )
    first_layer = ResidualBlock(
        first_layer,
        IdentityLayer(channels[1], channels[1])
        if channels[0] == channels[1]
        else None,
    )

    return first_conv, first_layer


def static_exit_stage_layers_builder(
    in_channels,
    out_channels,
    kernel_sizes,
    act,
    dropout_rate,
    n_classes
):

    final_expand_layer = ConvLayer(
        in_channels=in_channels,
        out_channels=out_channels[0],
        kernel_size=kernel_sizes[0],
        act_func=act[0],
        use_bn=True,
        ops_order="weight_bn_act"
    )
    feature_mix_layer = ConvLayer(
        in_channels=out_channels[0],
        out_channels=out_channels[1],
        kernel_size=kernel_sizes[1],
        act_func=act[1],
        bias=False,
        use_bn=False
    )

    classifier = LinearLayer(
        in_features=out_channels[1],
        out_features=n_classes,
        dropout_rate=dropout_rate)

    return final_expand_layer, feature_mix_layer, classifier


def static_body_stage_layers_builder(
    depth,
    in_channels,
    out_channels,
    kernel_size,
    expand_ratio,
    stride,
    act,
    use_se
):

    layers = []

    for i in range(depth):

        mb_conv = MBConvLayer(
            in_channels=in_channels if i == 0 else out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride if i == 0 else 1,
            expand_ratio=expand_ratio,
            act_func=act,
            use_se=use_se,
        )
        if i != 0:
            shortcut = IdentityLayer(out_channels, out_channels)
        else:
            shortcut = None

        layers.append(ResidualBlock(mb_conv, shortcut))

    return layers


def static_dense_body_stage_layers_builder(
    depth,
    in_channels,
    out_channels,
    kernel_size,
    expand_ratio,
    stride,
    act,
    use_se
):
    return static_body_stage_layers_builder(
        depth,
        in_channels,
        out_channels,
        kernel_size,
        expand_ratio,
        stride,
        act,
        use_se
    )


def static_parallel_body_stage_layers_builder(
    depth,
    in_channels,
    out_channels,
    kernel_size,
    expand_ratio,
    stride,
    act,
    use_se
):

    mb_conv_list = static_body_stage_layers_builder(
        depth,
        in_channels,
        out_channels,
        kernel_size,
        expand_ratio,
        stride,
        act,
        use_se
    )

    conv1x1_list = conv1x1_layers_builder(
        depth,
        in_channels,
        out_channels,
        stride,
        act
    )

    # attention_conv_list = attention_conv_layers_builder(
    #     depth,
    #     in_channels,
    #     out_channels,
    #     kernel_size,
    #     stride
    # )

    transformation_layers_list = transformation_layers_builder(
        depth,
        in_channels,
        out_channels,
        stride,
        act
    )

    layers = []
    for i in range(depth):
        # layers.append([mb_conv_list[i], conv1x1_list[i], attention_conv_list[i]])
        layers.append([mb_conv_list[i], conv1x1_list[i], transformation_layers_list[i]])

    return layers


def static_dense_parallel_body_stage_layers_builder(
    depth,
    in_channels,
    out_channels,
    kernel_size,
    expand_ratio,
    stride,
    act,
    use_se
):

    return static_parallel_body_stage_layers_builder(
        depth,
        in_channels,
        out_channels,
        kernel_size,
        expand_ratio,
        stride,
        act,
        use_se
    )


def conv1x1_layers_builder(
    depth,
    in_channels,
    out_channels,
    stride,
    act
):
    layers = []

    for i in range(depth):

        conv1x1 = ConvLayer(
            in_channels=in_channels if i == 0 else out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride if i == 0 else 1,
            act_func=act,
        )

        layers.append(conv1x1)

    return layers


def attention_conv_layers_builder(
    depth,
    in_channels,
    out_channels,
    kernel_size,
    stride,
):
    layers = []

    for i in range(depth):
        attention_conv = AttentionConv(
            in_channels=in_channels if i == 0 else out_channels,
            out_channels=out_channels,
            stride=stride if i == 0 else 1,
        )

        layers.append(attention_conv)

    return layers


def transformation_layers_builder(
    depth,
    in_channels,
    out_channels,
    stride,
    act
):

    layers = []

    for i in range(depth):
        transformation_layer = TransformationLayer(
            in_channels=in_channels if i == 0 else out_channels,
            out_channels=out_channels,
            stride=stride if i == 0 else 1,
            act_funct=act
        )

        layers.append(transformation_layer)

    return layers
