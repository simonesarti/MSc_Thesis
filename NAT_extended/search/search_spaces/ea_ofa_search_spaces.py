import random
from collections import OrderedDict

import numpy as np

from NAT_extended.search.search_spaces import SearchSpace
from OFA_mbv3_extended.networks.nets.my_networks import get_valid_nw_keys


def push_zeros_to_end(arr):
    n = len(arr)
    count = 0

    for i in range(n):
        if arr[i] != 0:
            arr[count] = arr[i]
            count += 1

    while count < n:
        arr[count] = 0
        count += 1


def fill_zeros_to_end(arr):
    zero_found = False
    for idx in range(len(arr)):
        if zero_found:
            arr[idx] = 0
        else:
            if arr[idx] == 0:
                zero_found = True


def sample_from_distribution(categories, distributions: list = None, probabilities: list = None):

    if distributions:
        # sample from a particular distribution
        # distribution = [(distribution_pointer, shape_params, sse), ...]
        assert len(distributions) == len(categories), "one distribution required per variable"

        x = []
        for valid_options, (distribution, params, _) in zip(categories, distributions):
            # Separate parts of parameters
            arg = params[:-2]
            loc = params[-2]
            scale = params[-1]

            try:
                _x = distribution.rvs(
                    *arg, loc=loc, scale=scale, size=1) if arg else distribution.rvs(loc=loc, scale=scale, size=1)
                _x = round(_x[0])

                if _x < min(valid_options):
                    _x = min(valid_options)

                if _x > max(valid_options):
                    _x = max(valid_options)

            except OverflowError:
                _x = random.choice(valid_options)

            x.append(_x)

    elif probabilities:
        # sample with specified probabilities for each variable
        # probabilities = [ [p1_lb, ..., p1_ub], [p2_lb, ..., p2_ub], .... ]
        assert len(probabilities) == len(categories), "one probability element required per variable"
        x = []
        for options, prob in zip(categories, probabilities):
            if prob is None:
                x.append(random.choice(options))    # use None where want uniform distribution for the variable
            else:
                assert len(options) == len(prob), "one probability must be specified for each variable value"
                x.append(random.choices(options, prob)[0])

    else:
        # uniform random sampling
        x = [random.choice(options) for options in categories]

    x = np.array(x).astype(int)

    return x


""" general super-classes """


class OFAMobileNetV3SearchSpace(SearchSpace):

    def __init__(
            self,
            image_scale_list,
            feature_encoding='one-hot',
            **kwargs
    ):
        super().__init__(**kwargs)

        assert isinstance(image_scale_list, list) or isinstance(image_scale_list, int)
        self.image_scale_list = image_scale_list if isinstance(image_scale_list, list) else [image_scale_list]

        self.ks_list = [3, 5, 7]
        self.depth_list = [2, 3, 4]
        self.expand_ratio_list = [3, 4, 6]
        self.width_mult_list = [1.0, 1.2]
        self.feature_encoding = feature_encoding

        # create the mappings between decision variables (genotype) and subnet architectural string (phenotype)
        self.str2var_mapping = OrderedDict()
        self.var2str_mapping = OrderedDict()
        self.str2var_mapping['skip'] = 0

        # select how to repair the array
        self.repair_mode = "push"

        # child class dependent values
        self.n_var = None
        self.lb = None
        self.ub = None
        self.categories = None
        self.stage_layer_indices = None

    def assign_mappings(self):
        raise NotImplementedError

    def set_categories(self, n_var, lb, ub, categories, stage_indexes):

        self.n_var = n_var
        self.lb = lb
        self.ub = ub
        self.categories = categories
        self.stage_layer_indices = stage_indexes

    def _sample(self, distributions: list = None, probabilities: list = None, subnet_str=True):

        x = sample_from_distribution(self.categories, distributions, probabilities)

        # repair, in case of zeroes in the middle of the stage
        x = self.repair(x)

        if subnet_str:
            return self._decode(x)

        else:
            return x

    def repair(self, x):

        for indices in self.stage_layer_indices:
            subarray_start_idx = indices[0]
            subarray_end_idx = indices[-1] + 1
            sub_array = x[subarray_start_idx:subarray_end_idx]

            if self.repair_mode == "push":
                push_zeros_to_end(sub_array)
            elif self.repair_mode == "fill":
                fill_zeros_to_end(sub_array)
            else:
                raise ValueError("repair mode can only be one of push/fill")

            for i in range(len(sub_array)):
                x[subarray_start_idx + i] = sub_array[i]

        return x

    def _features(self, X):
        # X should be a 2D matrix with each row being a decision variable vector
        if self.feat_enc is None:
            # in case the feature encoder is not initialized
            if self.feature_encoding == 'one-hot':
                from sklearn.preprocessing import OneHotEncoder
                self.feat_enc = OneHotEncoder(categories=self.categories).fit(X)
            elif self.feature_encoding == 'integer':
                pass
            else:
                raise NotImplementedError

        if self.feature_encoding == 'one-hot':
            return self.feat_enc.transform(X).toarray()
        if self.feature_encoding == 'integer':
            return X

    def var2str(self, v):
        if v > 0:
            return self.var2str_mapping[v]
        else:
            max_value = max(list(self.var2str_mapping.keys()))
            return self.var2str_mapping[random.randint(1, max_value)]

    def set_repair_mode(self, repair_mode):

        if repair_mode not in ["push", "fill"]:
            raise ValueError("repair mode can only be one of push/fill")
        else:
            self.repair_mode = repair_mode

    def set_initial_probs(self):

        idx = self.n_var - 20 + 2     # first 3rd layer
        skip = 4    # len of stage
        initial_probs = [None]*self.n_var

        max_value = max(list(self.var2str_mapping.keys()))
        zero_p = 1/3
        other_p = (1 - zero_p) / max_value

        for _ in range(5):
            initial_probs[idx] = [zero_p] + [other_p]*max_value
            initial_probs[idx + 1] = [zero_p] + [other_p]*max_value
            idx += skip

        return initial_probs

    """ implemented in subclasses"""

    @property
    def name(self):
        raise NotImplementedError

    # def str2var(self, ks, e, ...):
    #    raise NotImplementedError

    def _encode(self, subnet_str):
        raise NotImplementedError

    def _decode(self, x):
        raise NotImplementedError


class OFAMobileNetV3NarrowSearchSpace(OFAMobileNetV3SearchSpace):

    def __init__(
            self,
            image_scale_list,
            feature_encoding='one-hot',
            **kwargs
    ):
        super(OFAMobileNetV3NarrowSearchSpace, self).__init__(image_scale_list, feature_encoding, **kwargs)
        self.assign_mappings()

    def assign_mappings(self):
        increment = 1
        for e in self.expand_ratio_list:
            for ks in self.ks_list:
                self.str2var_mapping[f'ks@{ks}_e@{e}'] = increment
                self.var2str_mapping[increment] = (ks, e)
                increment += 1

    def str2var(self, ks, e):
        return self.str2var_mapping[f'ks@{ks}_e@{e}']

    """ implemented in subclasses"""

    @property
    def name(self):
        raise NotImplementedError

    def _encode(self, subnet_str):
        raise NotImplementedError

    def _decode(self, x):
        raise NotImplementedError


class OFAMobileNetV3WideSearchSpace(OFAMobileNetV3SearchSpace):

    def __init__(
            self,
            image_scale_list,
            feature_encoding,
            **kwargs
    ):
        super(OFAMobileNetV3WideSearchSpace, self).__init__(image_scale_list, feature_encoding, **kwargs)
        self.net_width_list = [1, 2, 3]
        self.assign_mappings()

    def assign_mappings(self):
        increment = 1
        for nw in get_valid_nw_keys(self.net_width_list):
            for e in self.expand_ratio_list:
                for ks in self.ks_list:
                    self.str2var_mapping[f'ks@{ks}_e@{e}_nw@{nw}'] = increment
                    self.var2str_mapping[increment] = (ks, e, nw)
                    increment += 1

    def str2var(self, ks, e, nw):
        return self.str2var_mapping[f'ks@{ks}_e@{e}_nw@{nw}']

    """ implemented in subclasses"""

    @property
    def name(self):
        raise NotImplementedError

    def _encode(self, subnet_str):
        raise NotImplementedError

    def _decode(self, x):
        raise NotImplementedError


""" classes used """


class OFAMobileNetV3BaseSpace(OFAMobileNetV3NarrowSearchSpace):

    def __init__(
        self,
        image_scale_list,
        feature_encoding='one-hot',
        **kwargs
    ):

        super(OFAMobileNetV3BaseSpace, self).__init__(
            image_scale_list,
            feature_encoding,
            **kwargs
        )

        n_var = 22

        # set lower bound to 1 for layer if it must exist, how many must exist depend on depth list min value
        lb_stages = [0, 0, 0, 0]
        for i in range(min(self.depth_list)):
            lb_stages[i] = 1

        max_value = max(list(self.var2str_mapping.keys()))

        lb = [0] + [0] + lb_stages * 5
        ub = [len(self.image_scale_list) - 1] + [len(self.width_mult_list) - 1] + [max_value] * 20

        # create the categories for each variable
        categories = [list(range(a, b + 1)) for a, b in zip(lb, ub)]

        stage_layer_indices = [list(range(2, n_var))[i:i + max(self.depth_list)]
                               for i in range(0, 20, max(self.depth_list))]

        self.set_categories(n_var, lb, ub, categories, stage_layer_indices)

        # assign sampling probabilities to some variables to have balanced archive in terms of stages depths
        self.initial_probs = self.set_initial_probs()

    @property
    def name(self):
        return 'b_ofamobilenetv3_ss'

    def _encode(self, subnet_str):
        # a sample subnet string
        # {'r' : 224,
        #  'w' : 1.2,
        #  'ks': [7, 7, 7, 7, 7, 3, 5, 3, 3, 5, 7, 3, 5, 5, 3, 3, 3, 3, 3, 5],
        #  'e' : [4, 6, 4, 6, 6, 6, 6, 6, 3, 4, 4, 4, 6, 4, 4, 3, 3, 6, 3, 4],
        #  'd' : [2, 2, 3, 4, 2]}

        x = [0] * self.n_var
        x[0] = np.where(subnet_str['r'] == np.array(self.image_scale_list))[0][0]
        x[1] = np.where(subnet_str['w'] == np.array(self.width_mult_list))[0][0]

        for indices, d in zip(self.stage_layer_indices, subnet_str['d']):
            for i in range(d):
                idx = indices[i]
                x[idx] = self.str2var(subnet_str['ks'][idx - 2], subnet_str['e'][idx - 2])
        return x

    def _decode(self, x):
        # a sample decision variable vector x corresponding to the above subnet string
        # [(image scale) 4,
        #  (width mult)  0,
        #  (layers)      8, 9, 5, 5, 6, 2, 3, 6, 6, 1, 4, 0, 1, 2, 2, 3, 9, 5, 8, 1]

        ks_list, expand_ratio_list, depth_list = [], [], []
        for indices in self.stage_layer_indices:
            d = len(indices)
            for idx in indices:
                ks, e = self.var2str(x[idx])
                ks_list.append(ks)
                expand_ratio_list.append(e)
                if x[idx] < 1:
                    d -= 1
            depth_list.append(d)

        return {
            'r': self.image_scale_list[x[0]],
            'w': self.width_mult_list[x[1]],
            'ks': ks_list,
            'e': expand_ratio_list,
            'd': depth_list
        }


class OFAMobileNetV3NetDepthSearchSpace(OFAMobileNetV3NarrowSearchSpace):

    def __init__(
            self,
            image_scale_list,
            feature_encoding='one-hot',
            **kwargs
    ):

        super(OFAMobileNetV3NetDepthSearchSpace, self).__init__(
            image_scale_list,
            feature_encoding,
            **kwargs
        )

        self.net_depth_list = [1, 2, 3, 4, 5]

        n_var = 23

        max_value = max(list(self.var2str_mapping.keys()))

        # set lower bound to 1 for layer if it must exist, how many must exist depend on depth list min value
        lb_stages = [0, 0, 0, 0]
        for i in range(min(self.depth_list)):
            lb_stages[i] = 1

        lb = [0] + [0] + [0] + lb_stages * 5
        ub = [len(self.image_scale_list) - 1] + [len(self.width_mult_list) - 1] + [len(self.net_depth_list) - 1] + [
            max_value] * 20

        # create the categories for each variable
        categories = [list(range(a, b + 1)) for a, b in zip(lb, ub)]

        stage_layer_indices = [list(range(3, n_var))[i:i + max(self.depth_list)]
                               for i in range(0, 20, max(self.depth_list))]

        self.set_categories(n_var, lb, ub, categories, stage_layer_indices)

        # assign sampling probabilities to some variables to have balanced archive in terms of stages depths
        self.initial_probs = self.set_initial_probs()

    @property
    def name(self):
        return 'nd_ofamobilenetv3_ss'

    def _encode(self, subnet_str):
        # a sample subnet string
        # {'r' : 224,
        #  'w' : 1.2,
        #  'nd': 2,
        #  'ks': [7, 7, 7, 7, 7, 3, 5, 3, 3, 5, 7, 3, 5, 5, 3, 3, 3, 3, 3, 5],
        #  'e' : [4, 6, 4, 6, 6, 6, 6, 6, 3, 4, 4, 4, 6, 4, 4, 3, 3, 6, 3, 4],
        #  'd' : [2, 2, 3, 4, 2]}

        x = [0] * self.n_var
        x[0] = np.where(subnet_str['r'] == np.array(self.image_scale_list))[0][0]
        x[1] = np.where(subnet_str['w'] == np.array(self.width_mult_list))[0][0]
        x[2] = np.where(subnet_str['nd'] == np.array(self.net_depth_list))[0][0]

        for indices, d in zip(self.stage_layer_indices, subnet_str['d']):
            for i in range(d):
                idx = indices[i]
                x[idx] = self.str2var(subnet_str['ks'][idx - 3], subnet_str['e'][idx - 3])
        return x

    def _decode(self, x):
        # a sample decision variable vector x corresponding to the above subnet string
        # [(image scale) 4,
        #  (width mult)  0,
        #  (net depth)   2,
        #  (layers)      8,9,5,5, 6,2,3,6, 6,1,4,0, 2,3,4,0, 1,7,0,0]

        ks_list, expand_ratio_list, depth_list = [], [], []
        for indices in self.stage_layer_indices:
            d = len(indices)
            for idx in indices:
                ks, e = self.var2str(x[idx])
                ks_list.append(ks)
                expand_ratio_list.append(e)
                if x[idx] < 1:
                    d -= 1
            depth_list.append(d)

        return {
            'r': self.image_scale_list[x[0]],
            'w': self.width_mult_list[x[1]],
            'nd': self.net_depth_list[x[2]],
            'ks': ks_list,
            'e': expand_ratio_list,
            'd': depth_list
        }


class OFAMobileNetV3ParallelSearchSpace(OFAMobileNetV3WideSearchSpace):

    def __init__(
        self,
        image_scale_list,
        feature_encoding='one-hot',
        **kwargs
    ):

        super(OFAMobileNetV3ParallelSearchSpace, self).__init__(
            image_scale_list,
            feature_encoding,
            **kwargs
        )

        n_var = 22

        # set lower bound to 1 for layer if it must exist, how many must exist depend on depth list min value
        lb_stages = [0, 0, 0, 0]
        for i in range(min(self.depth_list)):
            lb_stages[i] = 1

        max_value = max(list(self.var2str_mapping.keys()))

        lb = [0] + [0] + lb_stages * 5
        ub = [len(self.image_scale_list) - 1] + [len(self.width_mult_list) - 1] + [max_value] * 20

        # create the categories for each variable
        categories = [list(range(a, b + 1)) for a, b in zip(lb, ub)]

        stage_layer_indices = [list(range(2, n_var))[i:i + max(self.depth_list)]
                               for i in range(0, 20, max(self.depth_list))]

        self.set_categories(n_var, lb, ub, categories, stage_layer_indices)

        # assign sampling probabilities to some variables to have balanced archive in terms of stages depths
        self.initial_probs = self.set_initial_probs()

    @property
    def name(self):
        return "p_ofamobilenetv3_ss"

    def _encode(self, subnet_str):
        # a sample subnet string
        # {'r' : 224,
        #  'w' : 1.2,
        #  'ks': [7, 7, 7, 7, 7, 3, 5, 3, 3, 5, 7, 3, 5, 5, 3, 3, 3, 3, 3, 5],
        #  'e' : [4, 6, 4, 6, 6, 6, 6, 6, 3, 4, 4, 4, 6, 4, 4, 3, 3, 6, 3, 4],
        #  'nw' : [1, 3, 5, 2, 5, 7, 4, 2, 4, 1, 5, 3, 4, 1, 3, 5, 2, 6, 2, 1],
        #  'd' : [2, 2, 3, 4, 2]}

        x = [0] * self.n_var
        x[0] = np.where(subnet_str['r'] == np.array(self.image_scale_list))[0][0]
        x[1] = np.where(subnet_str['w'] == np.array(self.width_mult_list))[0][0]

        for indices, d in zip(self.stage_layer_indices, subnet_str['d']):
            for i in range(d):
                idx = indices[i]
                x[idx] = self.str2var(subnet_str['ks'][idx - 2], subnet_str['e'][idx - 2], subnet_str['nw'][idx - 2])
        return x

    def _decode(self, x):
        # a sample decision variable vector x corresponding to the above subnet string
        # [(image scale) 4,
        #  (width mult)  0,
        #  (layers)      8, 11, 5, 5, 14, 2, 3, 6, 6, 1, 12, 0, 1, 2, 2, 0, 9, 5, 8, 1]

        ks_list, expand_ratio_list, depth_list, net_widths = [], [], [], []
        for indices in self.stage_layer_indices:
            d = len(indices)
            for idx in indices:
                ks, e, nw = self.var2str(x[idx])
                ks_list.append(ks)
                expand_ratio_list.append(e)
                net_widths.append(nw)
                if x[idx] < 1:
                    d -= 1
            depth_list.append(d)

        return {
            'r': self.image_scale_list[x[0]],
            'w': self.width_mult_list[x[1]],
            'ks': ks_list,
            'e': expand_ratio_list,
            'nw': net_widths,
            'd': depth_list
        }


class OFAMobileNetV3NetDepthParallelSearchSpace(OFAMobileNetV3WideSearchSpace):
    def __init__(
            self,
            image_scale_list,
            feature_encoding='one-hot',
            **kwargs
    ):
        super(OFAMobileNetV3NetDepthParallelSearchSpace, self).__init__(
            image_scale_list,
            feature_encoding,
            **kwargs
        )
        self.net_depth_list = [1, 2, 3, 4, 5]

        n_var = 23

        # set lower bound to 1 for layer if it must exist, how many must exist depend on depth list min value
        lb_stages = [0, 0, 0, 0]
        for i in range(min(self.depth_list)):
            lb_stages[i] = 1

        max_value = max(list(self.var2str_mapping.keys()))

        lb = [0] + [0] + [0] + lb_stages * 5
        ub = [len(self.image_scale_list) - 1] + [len(self.width_mult_list) - 1] + [len(self.net_depth_list) - 1] + [
            max_value] * 20

        # create the categories for each variable
        categories = [list(range(a, b + 1)) for a, b in zip(lb, ub)]

        stage_layer_indices = [list(range(3, n_var))[i:i + max(self.depth_list)]
                               for i in range(0, 20, max(self.depth_list))]

        self.set_categories(n_var, lb, ub, categories, stage_layer_indices)

        # assign sampling probabilities to some variables to have balanced archive in terms of stages depths
        self.initial_probs = self.set_initial_probs()

    @property
    def name(self):
        return "nd_p_ofamobilenetv3_ss"

    def _encode(self, subnet_str):
        # a sample subnet string
        # {'r' : 224,
        #  'w' : 1.2,
        #  'nd': 2,
        #  'ks': [7, 7, 7, 7, 7, 3, 5, 3, 3, 5, 7, 3, 5, 5, 3, 3, 3, 3, 3, 5],
        #  'e' : [4, 6, 4, 6, 6, 6, 6, 6, 3, 4, 4, 4, 6, 4, 4, 3, 3, 6, 3, 4],
        #  'nw' : [1, 3, 5, 2, 5, 7, 4, 2, 4, 1, 5, 3, 4, 1, 3, 5, 2, 6, 2, 1],
        #  'd' : [2, 2, 3, 4, 2]}

        x = [0] * self.n_var
        x[0] = np.where(subnet_str['r'] == np.array(self.image_scale_list))[0][0]
        x[1] = np.where(subnet_str['w'] == np.array(self.width_mult_list))[0][0]
        x[2] = np.where(subnet_str['nd'] == np.array(self.net_depth_list))[0][0]

        for indices, d in zip(self.stage_layer_indices, subnet_str['d']):
            for i in range(d):
                idx = indices[i]
                x[idx] = self.str2var(subnet_str['ks'][idx - 3], subnet_str['e'][idx - 3], subnet_str['nw'][idx - 3])
        return x

    def _decode(self, x):
        # a sample decision variable vector x corresponding to the above subnet string
        # [(image scale) 4,
        #  (width mult)  0,
        #  (net_depth)   2,
        #  (layers)      8, 11, 5, 5, 14, 2, 3, 6, 6, 1, 12, 0, 1, 2, 2, 0, 9, 5, 8, 1]

        ks_list, expand_ratio_list, depth_list, net_widths = [], [], [], []
        for indices in self.stage_layer_indices:
            d = len(indices)
            for idx in indices:
                ks, e, nw = self.var2str(x[idx])
                ks_list.append(ks)
                expand_ratio_list.append(e)
                net_widths.append(nw)
                if x[idx] < 1:
                    d -= 1
            depth_list.append(d)

        return {
            'r': self.image_scale_list[x[0]],
            'w': self.width_mult_list[x[1]],
            'nd': self.net_depth_list[x[2]],
            'ks': ks_list,
            'e': expand_ratio_list,
            'nw': net_widths,
            'd': depth_list
        }



