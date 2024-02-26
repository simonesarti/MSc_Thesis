def get_search_space(net_name, search_type, image_scale_list, feature_encoding='one-hot'):

    if search_type == "ea":

        if net_name in ["OFAMobileNetV3", "SE_B_OFAMobileNetV3", "SE_D_OFAMobileNetV3"]:
            from .ea_ofa_search_spaces import OFAMobileNetV3BaseSpace
            return OFAMobileNetV3BaseSpace(image_scale_list, feature_encoding)

        elif net_name in ["SE_P_OFAMobileNetV3", "SE_DP_OFAMobileNetV3"]:
            from .ea_ofa_search_spaces import OFAMobileNetV3ParallelSearchSpace
            return OFAMobileNetV3ParallelSearchSpace(image_scale_list, feature_encoding)

        elif net_name in ["EE_B_OFAMobileNetV3", "EE_D_OFAMobileNetV3"]:
            from .ea_ofa_search_spaces import OFAMobileNetV3NetDepthSearchSpace
            return OFAMobileNetV3NetDepthSearchSpace(image_scale_list, feature_encoding)

        elif net_name in ["EE_P_OFAMobileNetV3", "EE_DP_OFAMobileNetV3"]:
            from .ea_ofa_search_spaces import OFAMobileNetV3NetDepthParallelSearchSpace
            return OFAMobileNetV3NetDepthParallelSearchSpace(image_scale_list, feature_encoding)

        else:
            raise NotImplementedError

    else:
        raise NotImplementedError
