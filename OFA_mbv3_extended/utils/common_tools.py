import os

import numpy as np

__all__ = [
    "rename_subfolders",
    "get_input_tensor",
    "check_list_values_allowed",
    "check_value_allowed",
    "partition_list",
    "reshape_list_axis0"
]


def rename_subfolders(root_dir, old_new_names_dict):
    old_names = []
    for it in os.scandir(root_dir):
        if it.is_dir():
            old_name = os.path.split(it.path)[-1]
            old_names.append(old_name)

    for old_name in old_names:
        old_path = os.path.join(root_dir, old_name)
        new_name = old_new_names_dict[old_name]
        new_path = os.path.join(root_dir, new_name)
        os.rename(old_path, new_path)


def get_input_tensor(curr_depth, tensors_list):
    # print(f"depth {curr_depth},tensor_list {tensors_list}")
    input_tensor = tensors_list[curr_depth]
    if 0 <= curr_depth <= 2:
        return input_tensor
    else:
        first_tensor_idx = 1
        last_tensor_idx = curr_depth-2
        tensors_to_sum = tensors_list[first_tensor_idx:last_tensor_idx+1]
        for tts in tensors_to_sum:
            input_tensor += tts

    # print(f"returning {input_tensor}\n")
    return input_tensor


def check_list_values_allowed(l, allowed_values):
    for elem in l:
        check_value_allowed(elem, allowed_values)


def check_value_allowed(value, allowed_values):
    if value not in allowed_values:
        raise ValueError(f"{value} not in allowed values {allowed_values}")


def partition_list(l, n_partitions):

    split_array = np.array_split(l, n_partitions)
    l = [list(array) for array in split_array]

    l = [[elem.item() for elem in list_elem] for list_elem in l]

    return l


def reshape_list_axis0(l):
    result = []
    for i in range(len(l[0])):
        dim0_sublist = []
        for sublist in l:
            dim0_sublist.append(sublist[i])
        result.append(dim0_sublist)
    return result







