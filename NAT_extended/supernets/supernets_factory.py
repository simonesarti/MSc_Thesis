from OFA_mbv3_extended.networks.nets.my_networks import get_net_by_name


def get_supernet(net_name, n_classes, dropout_rate, search_space):

    net_class = get_net_by_name(net_name)

    if net_name in ["OFAMobileNetV3", "SE_B_OFAMobileNetV3", "SE_D_OFAMobileNetV3"]:
        from .supernet import SE_Narrow_GenOFAMobileNetV3
        supernet = SE_Narrow_GenOFAMobileNetV3(
            n_classes=n_classes,
            dropout_rate=dropout_rate,
            net_class=net_class,
            image_scale_list=search_space.image_scale_list,
            width_mult_list=search_space.width_mult_list,
            ks_list=search_space.ks_list,
            expand_ratio_list=search_space.expand_ratio_list,
            depth_list=search_space.depth_list
        )

    elif net_name in ["SE_P_OFAMobileNetV3", "SE_DP_OFAMobileNetV3"]:
        from .supernet import SE_Wide_GenOFAMobileNetV3
        supernet = SE_Wide_GenOFAMobileNetV3(
            n_classes=n_classes,
            dropout_rate=dropout_rate,
            net_class=net_class,
            image_scale_list=search_space.image_scale_list,
            width_mult_list=search_space.width_mult_list,
            ks_list=search_space.ks_list,
            expand_ratio_list=search_space.expand_ratio_list,
            depth_list=search_space.depth_list,
            net_width_list=search_space.net_width_list
        )

    elif net_name in ["EE_B_OFAMobileNetV3", "EE_D_OFAMobileNetV3"]:
        from .supernet import EE_Narrow_GenOFAMobileNetV3
        supernet = EE_Narrow_GenOFAMobileNetV3(
            n_classes=n_classes,
            dropout_rate=dropout_rate,
            net_class=net_class,
            image_scale_list=search_space.image_scale_list,
            width_mult_list=search_space.width_mult_list,
            ks_list=search_space.ks_list,
            expand_ratio_list=search_space.expand_ratio_list,
            depth_list=search_space.depth_list,
            net_depth_list=search_space.net_depth_list
        )

    elif net_name in ["EE_P_OFAMobileNetV3", "EE_DP_OFAMobileNetV3"]:
        from .supernet import EE_Wide_GenOFAMobileNetV3
        supernet = EE_Wide_GenOFAMobileNetV3(
            n_classes=n_classes,
            dropout_rate=dropout_rate,
            net_class=net_class,
            image_scale_list=search_space.image_scale_list,
            width_mult_list=search_space.width_mult_list,
            ks_list=search_space.ks_list,
            expand_ratio_list=search_space.expand_ratio_list,
            depth_list=search_space.depth_list,
            net_depth_list=search_space.net_depth_list,
            net_width_list=search_space.net_width_list
        )

    else:
        raise NotImplementedError

    return supernet
