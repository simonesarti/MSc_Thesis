from ofa.utils.layers import LinearLayer
from ofa.utils.layers import ResidualBlock
from ofa.utils.my_modules import MyModule
from ofa.utils.pytorch_modules import MyGlobalAvgPool2d
from torch import nn

from OFA_mbv3_extended.networks.layers.static_layers.stat_layers import ext_set_layer_from_config
from OFA_mbv3_extended.utils.common_tools import get_input_tensor

__all__ = [
    "MyStage",
    "StaticHeadStage",
    "StaticBodyStage",
    "StaticExitStage",
    "StaticDenseBodyStage",
    "StaticParallelBodyStage",
    "StaticDenseParallelBodyStage",
]


class MyStage(MyModule):

    def forward(self, x):
        raise NotImplementedError

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError


class StaticHeadStage(MyStage):

    def __init__(self, first_conv, first_layer):

        super(StaticHeadStage, self).__init__()

        self.first_conv = first_conv
        self.first_layer = first_layer

    def forward(self, x):

        x = self.first_conv(x)
        x = self.first_layer(x)
        return x

    @property
    def module_str(self):
        _str = ""
        _str += self.first_conv.module_str + "\n"
        _str += self.first_layer.module_str
        return _str

    @property
    def config(self):

        return {
            "name": StaticHeadStage.__name__,
            "first_conv": self.first_conv.config,
            "first_layer": self.first_layer.config
        }

    @staticmethod
    def build_from_config(config):
        first_conv = ext_set_layer_from_config(config["first_conv"])
        first_layer = ext_set_layer_from_config(config["first_layer"])
        head_stage = StaticHeadStage(first_conv, first_layer)

        return head_stage


class StaticBodyStage(MyStage):

    def __init__(self, layers):

        super(StaticBodyStage, self).__init__()

        self.layers = nn.ModuleList(layers)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x

    @property
    def module_str(self):
        _str = ""
        for layer in self.layers:
            _str += layer.module_str + "\n"

        return _str

    @property
    def config(self):

        return {
            "name": StaticBodyStage.__name__,
            "layers": [layer.config for layer in self.layers]
        }

    @staticmethod
    def build_from_config(config):

        layers = []
        for layer_config in config["layers"]:
            layers.append(ResidualBlock.build_from_config(layer_config))

        body_stage = StaticBodyStage(layers)

        return body_stage


class StaticExitStage(MyStage):

    def __init__(self, final_expand_layer, feature_mix_layer, classifier):

        super(StaticExitStage, self).__init__()

        self.final_expand_layer = final_expand_layer
        self.global_avg_pooling = MyGlobalAvgPool2d(keep_dim=True)
        self.feature_mix_layer = feature_mix_layer
        self.classifier = classifier

    def forward(self, x):
        x = self.final_expand_layer(x)
        x = self.global_avg_pooling(x)
        x = self.feature_mix_layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = ""
        _str += self.final_expand_layer.module_str + "\n"
        _str += self.global_avg_pooling.__repr__() + "\n"
        _str += self.feature_mix_layer.module_str + "\n"
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            "name": StaticExitStage.__name__,
            "final_expand_layer": self.final_expand_layer.config,
            "feature_mix_layer": self.feature_mix_layer.config,
            "classifier": self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        final_expand_layer = ext_set_layer_from_config(config["final_expand_layer"])
        feature_mix_layer = ext_set_layer_from_config(config["feature_mix_layer"])
        classifier = ext_set_layer_from_config(config["classifier"])
        exit_stage = StaticExitStage(final_expand_layer, feature_mix_layer, classifier)

        return exit_stage

    def reset_classifier(self, n_classes, dropout_rate=0.0):
        in_features = self.classifier.in_features
        self.classifier = LinearLayer(in_features, n_classes, dropout_rate=dropout_rate)


class StaticDenseBodyStage(StaticBodyStage):

    def __init__(self, layers):
        super(StaticDenseBodyStage, self).__init__(layers)

    def forward(self, x):

        depth = len(self.layers)
        y = [x]
        for d in range(depth):
            input_tensor = get_input_tensor(d, y)
            y.append(self.layers[d](input_tensor))

        return get_input_tensor(depth, y)

    # @property
    # def module_str(self):
    # ==> inherited from StaticBodyStage

    @property
    def config(self):

        return {
            "name": StaticDenseBodyStage.__name__,
            "layers": [layer.config for layer in self.layers]
        }

    @staticmethod
    def build_from_config(config):

        layers = []
        for layer_config in config["layers"]:
            layers.append(ResidualBlock.build_from_config(layer_config))

        body_stage = StaticDenseBodyStage(layers)

        return body_stage


class StaticParallelBodyStage(MyStage):

    def __init__(
            self,
            layers_blocks
    ):

        super(StaticParallelBodyStage, self).__init__()
        self.layers_blocks = nn.ModuleList()
        for group_of_parallel_layers in layers_blocks:
            self.layers_blocks.append(nn.ModuleList(group_of_parallel_layers))

    def forward(self, x):

        for group_of_parallel_layers in self.layers_blocks:
            parallel_results = []

            # pass input tensor x through all the parallel layers,
            # save the parallel results in y
            for layer in group_of_parallel_layers:
                parallel_results.append(layer(x))

            # compute the input to the next block of layers as the sum of the results of the parallel layers
            x = 0
            for res_tensor in parallel_results:
                x += res_tensor

        return x

    @property
    def module_str(self):

        _str = ""
        for group_of_parallel_layers in self.layers_blocks:
            for layer in group_of_parallel_layers:
                _str += layer.module_str + "--"
            _str += "\n"

        return _str

    @property
    def config(self):

        layers_blocks_configs = []
        for group_of_parallel_layers in self.layers_blocks:
            layers_configs = [layer.config for layer in group_of_parallel_layers]
            layers_blocks_configs.append(layers_configs)

        return {
            "name": StaticParallelBodyStage.__name__,
            "layers_blocks": layers_blocks_configs
        }

    @staticmethod
    def build_from_config(config):

        layers_blocks = []
        for group_of_parallel_layers_config in config["layers_blocks"]:
            group_of_parallel_layers = []
            for layer_config in group_of_parallel_layers_config:
                group_of_parallel_layers.append(ext_set_layer_from_config(layer_config))
            layers_blocks.append(group_of_parallel_layers)

        body_stage = StaticParallelBodyStage(layers_blocks)

        return body_stage


class StaticDenseParallelBodyStage(StaticParallelBodyStage):

    def __init__(
            self,
            layers_blocks
    ):

        super(StaticDenseParallelBodyStage, self).__init__(layers_blocks)

    def forward(self, x):

        depth = len(self.layers_blocks)
        y = [x]
        for d in range(depth):
            input_tensor = get_input_tensor(d, y)

            parallel_results = []
            # pass input tensor through all the parallel layers, save parallel results
            for layer in self.layers_blocks[d]:
                parallel_results.append(layer(input_tensor))

            # add the obtained tensors
            x = 0
            for res in parallel_results:
                x += res

            y.append(x)

        return get_input_tensor(depth, y)

    # @property
    # def module_str(self):
    # ==> inherited from StaticParallelBodyStage

    @property
    def config(self):
        layers_blocks_configs = []
        for group_of_parallel_layers in self.layers_blocks:
            layers_configs = [layer.config for layer in group_of_parallel_layers]
            layers_blocks_configs.append(layers_configs)

        return {
            "name": StaticDenseParallelBodyStage.__name__,
            "layers_blocks": layers_blocks_configs
        }

    @staticmethod
    def build_from_config(config):

        layers_blocks = []
        for group_of_parallel_layers_config in config["layers_blocks"]:
            group_of_parallel_layers = []
            for layer_config in group_of_parallel_layers_config:
                group_of_parallel_layers.append(ext_set_layer_from_config(layer_config))
            layers_blocks.append(group_of_parallel_layers)

        body_stage = StaticDenseParallelBodyStage(layers_blocks)

        return body_stage


