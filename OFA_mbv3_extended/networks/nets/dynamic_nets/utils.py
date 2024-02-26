from ofa.utils import val2list

from OFA_mbv3_extended.utils.common_tools import check_list_values_allowed, check_value_allowed, partition_list


def check_and_reformat_ks_e_d_helper(n_stages, ks, e, d, ks_list, expand_ratio_list, depth_list):

    stage_depth = max(depth_list)
    net_depth = n_stages * stage_depth

    ks_l = check_expand_single_list(ks, net_depth, ks_list)
    ks_l = partition_list(ks_l, n_stages)

    e_l = check_expand_single_list(e, net_depth, expand_ratio_list)
    e_l = partition_list(e_l, n_stages)

    d_l = check_expand_single_list(d, n_stages, depth_list)

    return ks_l, e_l, d_l


def check_expand_single_list(l, expected_len, allowed_values):

    if isinstance(l, list):

        if len(l) != expected_len:
            raise ValueError(f"Wrong size for list, expenced {expected_len}, got {len(l)}")

        for elem in l:
            if not isinstance(elem, int):
                raise ValueError("elements of the list must be integers")

        check_list_values_allowed(l, allowed_values)

    elif isinstance(l, int):
        check_value_allowed(l, allowed_values)
        l = val2list(l, expected_len)

    else:
        raise ValueError(f"expected list or int, got {type(l)}")

    return l


def stages_settings_to_list(dict_list):

    settings = {
        "ks": [],
        "e": [],
        "d": []
    }

    for dic in dict_list:
        settings["ks"] += dic["ks"]
        settings["e"] += dic["e"]
        settings["d"].append(dic["d"])

    return settings


def stages_settings_to_list_nw(dict_list):

    base_settings = stages_settings_to_list(dict_list)
    nw_settings = []

    for dic in dict_list:
        nw_settings += dic["nw"]

    settings = {
        **base_settings,
        "nw": nw_settings
    }

    return settings
