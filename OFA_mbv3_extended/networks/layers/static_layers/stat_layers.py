import torch
import torch.nn as nn
import torch.nn.init as init
from ofa.utils import get_same_padding, build_activation
from ofa.utils.layers import (
    ConvLayer,
    IdentityLayer,
    LinearLayer,
    ZeroLayer,
    MBConvLayer,
    ResidualBlock,
    # ResNetBottleneckBlock,
    # MultiHeadLinearLayer,
    # My2DLayer
)
from ofa.utils.my_modules import MyModule
from torch.nn import functional as F


def ext_set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    name2layer = {
        ConvLayer.__name__: ConvLayer,
        IdentityLayer.__name__: IdentityLayer,
        LinearLayer.__name__: LinearLayer,
        ZeroLayer.__name__: ZeroLayer,
        MBConvLayer.__name__: MBConvLayer,
        "MBInvertedConvLayer": MBConvLayer,
        ResidualBlock.__name__: ResidualBlock,
        ##########################################################
        AttentionConv.__name__: AttentionConv,
        SelfAttention.__name__: SelfAttention,
        TransformationLayer.__name__: TransformationLayer,
        #########################################################
        # MultiHeadLinearLayer.__name__: MultiHeadLinearLayer,
        # ResNetBottleneckBlock.__name__: ResNetBottleneckBlock,
        # DepthConvLayer.__name__: DepthConvLayer,
        # PoolingLayer.__name__: PoolingLayer
        # MobileInvertedResidualBlock.__name__: MobileInvertedResidualBlock

    }

    layer_name = layer_config.pop("name")
    layer = name2layer[layer_name]
    return layer.build_from_config(layer_config)


class TransformationLayer(MyModule):

    def __init__(self, in_channels, out_channels, stride=1, act_funct="relu"):
        super(TransformationLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.act_funct = act_funct

        if stride > 1:
            self.mp2d = nn.MaxPool2d(
                kernel_size=self.stride,
                stride=self.stride,
                ceil_mode=True
            )
        if in_channels != out_channels:
            self.conv1x1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1
            )

        self.transformation = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            build_activation(act_funct, inplace=True)
        )

    def forward(self, x):

        if self.stride > 1:
            x = self.mp2d(x)
        if self.in_channels != self.out_channels:
            x = self.conv1x1(x)

        x = self.transformation(x)
        return x

    @property
    def module_str(self):
        return "TransformationLayer{}to{}".format(self.in_channels, self.out_channels)

    @property
    def config(self):
        return {
            "name": TransformationLayer.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "stride": self.stride,
            "act_funct": self.act_funct
        }

    @staticmethod
    def build_from_config(config):
        return TransformationLayer(**config)


class AttentionConv(MyModule):

    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, groups=8, bias=False):
        super(AttentionConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = get_same_padding(kernel_size)
        self.groups = groups
        self.bias = bias

        assert out_channels % groups == 0

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()
        stride = 1  # only reduce spatial dimension with avg_pool after attention if needed

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, stride).unfold(3, self.kernel_size, stride)
        v_out = v_out.unfold(2, self.kernel_size, stride).unfold(3, self.kernel_size, stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        # reduce spatial dimension with avg_pool if stride > 1
        if self.stride != 1:
            padding = 0 if width % 2 == 0 else 1
            out = F.avg_pool2d(out, self.stride, self.stride, padding=padding)
        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)

    @property
    def module_str(self):
        _str = "ATTENTIONCONV"
        _str += "_%d" % self.out_channels
        return _str

    @property
    def config(self):
        return {
            "name": AttentionConv.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "groups": self.groups,
            "bias": self.bias
        }

    @staticmethod
    def build_from_config(config):
        return AttentionConv(**config)


class SelfAttention(MyModule):
    """ Self attention Layer """

    def __init__(self, in_channels, out_channels, stride, act_func):
        super(SelfAttention, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.act_func = act_func

        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()

        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X N X C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x N
        energy = torch.bmm(proj_query, proj_key)  # transpose check, B X N X N
        attention = self.softmax(energy)  # softmax(B X N X N)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # (B X C X N) * (B X N X N) = B X C X N
        out = out.view(m_batchsize, C, width, height)  # B X C X W X H

        out = self.gamma * out + x

        if self.stride != 1:
            padding = 0 if width % 2 == 0 else 1
            out = F.avg_pool2d(out, self.stride, self.stride, padding=padding)
        return out
        # return out, attention

    @property
    def module_str(self):
        _str = "SELF_ATTENTION_"
        _str += "%d_" % self.out_channels
        _str += self.act_func.upper()
        return _str

    def config(self):
        return {
            "name": SelfAttention.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "stride": self.stride,
            "act_func": self.act_func,
        }

    @staticmethod
    def build_from_config(config):
        return SelfAttention(**config)


"""
class DepthConvLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, dilation=1, groups=1, bias=False, has_shuffle=False,
                 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act'):
        # default normal 3x3_DepthConv with bn and relu
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle

        super(DepthConvLayer, self).__init__(
            in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order,
        )

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict()
        weight_dict['depth_conv'] = nn.Conv2d(
            self.in_channels, self.in_channels, kernel_size=self.kernel_size, stride=self.stride, padding=padding,
            dilation=self.dilation, groups=self.in_channels, bias=False
        )
        weight_dict['point_conv'] = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=1, groups=self.groups, bias=self.bias
        )
        if self.has_shuffle and self.groups > 1:
            weight_dict['shuffle'] = ShuffleLayer(self.groups)
        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.dilation > 1:
            conv_str = '%dx%d_DilatedDepthConv' % (kernel_size[0], kernel_size[1])
        else:
            conv_str = '%dx%d_DepthConv' % (kernel_size[0], kernel_size[1])
        conv_str += '_O%d' % self.out_channels
        return conv_str

    @property
    def config(self):
        return {
            'name': DepthConvLayer.__name__,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
            'has_shuffle': self.has_shuffle,
            **super(DepthConvLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return DepthConvLayer(**config)


class PoolingLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 pool_type, kernel_size=2, stride=2,
                 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        self.pool_type = pool_type
        self.kernel_size = kernel_size
        self.stride = stride

        super(PoolingLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_op(self):
        if self.stride == 1:
            # same padding if `stride == 1`
            padding = get_same_padding(self.kernel_size)
        else:
            padding = 0

        weight_dict = OrderedDict()
        if self.pool_type == 'avg':
            weight_dict['pool'] = nn.AvgPool2d(
                self.kernel_size, stride=self.stride, padding=padding, count_include_pad=False
            )
        elif self.pool_type == 'max':
            weight_dict['pool'] = nn.MaxPool2d(self.kernel_size, stride=self.stride, padding=padding)
        else:
            raise NotImplementedError
        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        return '%dx%d_%sPool' % (kernel_size[0], kernel_size[1], self.pool_type.upper())

    @property
    def config(self):
        return {
            'name': PoolingLayer.__name__,
            'pool_type': self.pool_type,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            **super(PoolingLayer, self).config
        }

    @staticmethod
    def build_from_config(config):
        return PoolingLayer(**config)
"""


def drop_connect(inputs, training=False, drop_connect_rate=0.):
    """Apply drop connect."""
    if not training:
        return inputs

    keep_prob = 1 - drop_connect_rate
    random_tensor = keep_prob + torch.rand((inputs.size()[0], 1, 1, 1), dtype=inputs.dtype, device=inputs.device)
    random_tensor.floor_()  # binarize
    output = inputs.div(keep_prob) * random_tensor
    return output


class MobileInvertedResidualBlock(MyModule):

    def __init__(self, mobile_inverted_conv, shortcut, drop_connect_rate=0.0):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut
        self.drop_connect_rate = drop_connect_rate

    def forward(self, x):
        if self.mobile_inverted_conv is None or isinstance(self.mobile_inverted_conv, ZeroLayer):
            res = x
        elif self.shortcut is None or isinstance(self.shortcut, ZeroLayer):
            res = self.mobile_inverted_conv(x)
        else:
            # res = self.mobile_inverted_conv(x) + self.shortcut(x)
            res = self.mobile_inverted_conv(x)

            if self.drop_connect_rate > 0.:
                res = drop_connect(res, self.training, self.drop_connect_rate)

            res += self.shortcut(x)

        return res

    @property
    def module_str(self):
        return '(%s, %s)' % (
            self.mobile_inverted_conv.module_str if self.mobile_inverted_conv is not None else None,
            self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': MobileInvertedResidualBlock.__name__,
            'mobile_inverted_conv': self.mobile_inverted_conv.config if self.mobile_inverted_conv is not None else None,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    #
    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = ext_set_layer_from_config(config['mobile_inverted_conv'])
        shortcut = ext_set_layer_from_config(config['shortcut'])
        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut, drop_connect_rate=config['drop_connect_rate'])

    # where:
    # block_config['drop_connect_rate'] = drop_connect_rate * block_idx / len(config['blocks'])
    # blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))
    # in "build from config of NATnet", drop_connect_rate set there
