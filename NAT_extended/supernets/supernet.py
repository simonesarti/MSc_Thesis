import random

import numpy as np


class GenSupernet:

    def __init__(
            self,
            image_scale_list,
            width_mult_list,
            ks_list,
            expand_ratio_list,
            depth_list,
            engine
    ):

        self.image_scale_list = image_scale_list
        self.width_mult_list = width_mult_list
        self.ks_list = ks_list
        self.expand_ratio_list = expand_ratio_list
        self.depth_list = depth_list
        self.active_width_mult_idx = 0

        self.engine = engine

    def forward(self, x):
        return self.engine[self.active_width_mult_idx](x)

    def parameters(self):
        return self.engine[self.active_width_mult_idx].parameters()

    def cuda(self):
        for i in range(len(self.engine)):
            self.engine[i] = self.engine[i].cuda()

    def get_active_subnet(self, preserve_weight=True):
        return self.engine[self.active_width_mult_idx].get_active_subnet(preserve_weight)

    def load_state_dict(self, state_dict_list):
        for i, state_dict in enumerate(state_dict_list):
            self.engine[i].load_state_dict(state_dict)

    def sample_active_subnet(self):

        image_scale = random.choice(self.image_scale_list)
        wid_mult_idx = random.choice(range(len(self.width_mult_list)))
        sub_str = self.engine[wid_mult_idx].sample_active_subnet()

        self.active_width_mult_idx = wid_mult_idx

        return {
            'r': image_scale,
            'w': self.width_mult_list[wid_mult_idx],
            **sub_str
        }

    def set_active_subnet(self, w=None, ks=None, e=None, d=None, **kwargs):

        if w is None:
            wid_mult_idx = self.active_width_mult_idx
        else:
            wid_mult_idx = np.where(w == np.array(self.width_mult_list))[0][0]
            self.active_width_mult_idx = wid_mult_idx

        self.engine[wid_mult_idx].set_active_subnet(ks=ks, e=e, d=d, **kwargs)


class SE_Narrow_GenOFAMobileNetV3(GenSupernet):

    def __init__(
            self,
            image_scale_list,
            net_class,
            width_mult_list,
            ks_list,
            expand_ratio_list,
            depth_list,
            n_classes=200,
            dropout_rate=0
    ):

        engine = [
            net_class(
                n_classes=n_classes,
                dropout_rate=dropout_rate,
                width_mult=width_mult,
                ks_list=ks_list,
                expand_ratio_list=expand_ratio_list,
                depth_list=depth_list
            ) for width_mult in width_mult_list
        ]

        super(SE_Narrow_GenOFAMobileNetV3, self).__init__(
            image_scale_list=image_scale_list,
            width_mult_list=width_mult_list,
            ks_list=ks_list,
            expand_ratio_list=expand_ratio_list,
            depth_list=depth_list,
            engine=engine
        )


class EE_Narrow_GenOFAMobileNetV3(GenSupernet):

    def __init__(
            self,
            image_scale_list,
            net_class,
            width_mult_list,
            ks_list,
            expand_ratio_list,
            depth_list,
            net_depth_list,
            n_classes=200,
            dropout_rate=0
    ):

        self.net_depth_list = net_depth_list

        engine = [
            net_class(
                n_classes=n_classes,
                dropout_rate=dropout_rate,
                width_mult=width_mult,
                ks_list=ks_list,
                expand_ratio_list=expand_ratio_list,
                depth_list=depth_list,
                net_depth_list=net_depth_list
            ) for width_mult in width_mult_list
        ]

        super(EE_Narrow_GenOFAMobileNetV3, self).__init__(
            image_scale_list=image_scale_list,
            width_mult_list=width_mult_list,
            ks_list=ks_list,
            expand_ratio_list=expand_ratio_list,
            depth_list=depth_list,
            engine=engine
        )

    def get_active_all_exits_subnet(self, preserve_weight=True):
        return self.engine[self.active_width_mult_idx].get_all_exits_subnet(preserve_weight)

    def get_active_some_exits_subnet(self, exits_pos, preserve_weight=True):
        return self.engine[self.active_width_mult_idx].get_some_exits_subnet(exits_pos, preserve_weight)


class SE_Wide_GenOFAMobileNetV3(GenSupernet):

    def __init__(
            self,
            image_scale_list,
            net_class,
            width_mult_list,
            ks_list,
            expand_ratio_list,
            depth_list,
            net_width_list,
            n_classes=200,
            dropout_rate=0
    ):

        self.net_width_list = net_width_list

        engine = [
            net_class(
                n_classes=n_classes,
                dropout_rate=dropout_rate,
                width_mult=width_mult,
                ks_list=ks_list,
                expand_ratio_list=expand_ratio_list,
                depth_list=depth_list,
                net_width_list = net_width_list
            ) for width_mult in width_mult_list
        ]

        super(SE_Wide_GenOFAMobileNetV3, self).__init__(
            image_scale_list=image_scale_list,
            width_mult_list=width_mult_list,
            ks_list=ks_list,
            expand_ratio_list=expand_ratio_list,
            depth_list=depth_list,
            engine=engine
        )


class EE_Wide_GenOFAMobileNetV3(GenSupernet):

    def __init__(
            self,
            image_scale_list,
            net_class,
            width_mult_list,
            ks_list,
            expand_ratio_list,
            depth_list,
            net_depth_list,
            net_width_list,
            n_classes=200,
            dropout_rate=0
    ):
        self.net_depth_list = net_depth_list
        self.net_width_list = net_width_list

        engine = [
            net_class(
                n_classes=n_classes,
                dropout_rate=dropout_rate,
                width_mult=width_mult,
                ks_list=ks_list,
                expand_ratio_list=expand_ratio_list,
                depth_list=depth_list,
                net_depth_list=net_depth_list,
                net_width_list=net_width_list,
            ) for width_mult in width_mult_list
        ]

        super(EE_Wide_GenOFAMobileNetV3, self).__init__(
            image_scale_list=image_scale_list,
            width_mult_list=width_mult_list,
            ks_list=ks_list,
            expand_ratio_list=expand_ratio_list,
            depth_list=depth_list,
            engine=engine
        )

    def get_active_all_exits_subnet(self, preserve_weight=True):
        return self.engine[self.active_width_mult_idx].get_all_exits_subnet(preserve_weight)

    def get_active_some_exits_subnet(self, exits_pos, preserve_weight=True):
        return self.engine[self.active_width_mult_idx].get_some_exits_subnet(exits_pos, preserve_weight)


