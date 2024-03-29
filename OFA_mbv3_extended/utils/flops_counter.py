import torch
import torch.nn as nn
from ofa.utils.my_modules import MyConv2d
from ofa.utils.pytorch_modules import Hsigmoid, Hswish

multiply_adds = 1


def count_convNd(m, _, y):
    cin = m.in_channels

    kernel_ops = m.weight.size()[2] * m.weight.size()[3]
    ops_per_element = kernel_ops
    output_elements = y.nelement()

    # cout x oW x oH
    total_ops = cin * output_elements * ops_per_element // m.groups
    m.total_ops = torch.Tensor([int(total_ops)])


def count_linear(m, _, __):
    total_ops = m.in_features * m.out_features

    m.total_ops = torch.Tensor([int(total_ops)])


def count_relu(m, x, _):
    x = x[0]

    n_elements = x.numel()
    total_ops = n_elements

    m.total_ops = torch.Tensor([int(total_ops)])


def count_hsigmoid(m, x, _):
    x = x[0]
    
    n_elements = x.numel()
    total_ops = n_elements * 3
    
    m.total_ops = torch.Tensor([int(total_ops)])
    

def count_hswish(m, x, _):
    x = x[0]
    
    n_elements = x.numel()
    total_ops = n_elements * 4

    m.total_ops = torch.Tensor([int(total_ops)])


register_hooks = {
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    MyConv2d: count_convNd,
    ######################################
    nn.Linear: count_linear,
    ######################################
    nn.ReLU: count_relu,
    nn.ReLU6: count_relu,
    Hsigmoid: count_hsigmoid,
    Hswish: count_hswish,
    ######################################
    nn.Dropout: None,
    nn.Dropout2d: None,
    nn.Dropout3d: None,
    nn.BatchNorm2d: None,
}


def profile(model, input_size, custom_ops=None):
    handler_collection = []
    custom_ops = {} if custom_ops is None else custom_ops
    
    def add_hooks(m_):
        # Returns an iterator over immediate children modules.
        if len(list(m_.children())) > 0:
            return

        # Adds a buffer to the module
        # This is typically used to register a buffer that should not to be considered a model parameter. For example,
        # BatchNorm’s running_mean is not a parameter, but is part of the module’s state. Buffers, by default,
        # are persistent and will be saved alongside parameters. This behavior can be changed by setting persistent
        # to False. The only difference between a persistent buffer and a non-persistent buffer is that the latter will
        # not be a part of this module’s state_dict.
        m_.register_buffer('total_ops', torch.zeros(1))
        m_.register_buffer('total_params', torch.zeros(1))

        # Returns an iterator over module parameters.
        for p in m_.parameters():
            m_.total_params += torch.Tensor([p.numel()])

        m_type = type(m_)
        fn = None

        if m_type in custom_ops:
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
        else:
            # print("Not implemented for ", m_)
            pass

        if fn is not None:
            # print("Register FLOP counter for module %s" % str(m_))

            # Registers a forward hook on the module.The hook will be called every time after forward()
            # has computed an output.
            _handler = m_.register_forward_hook(fn)
            handler_collection.append(_handler)

    original_device = model.parameters().__next__().device
    training = model.training

    # Sets the module in evaluation mode.
    # This has any effect only on certain modules. See documentations of particular modules for
    # details of their behaviors in training/evaluation mode, if they are affected, e.g. Dropout, BatchNorm
    model.eval()

    # Applies fn recursively to every submodule (as returned by .children()) as well as self.
    # Typical use includes initializing the parameters of a model
    model.apply(add_hooks)

    x = torch.zeros(input_size).to(original_device)
    with torch.no_grad():
        model(x)

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue
        total_ops += m.total_ops
        total_params += m.total_params

    total_ops = total_ops.item()
    total_params = total_params.item()

    model.train(training).to(original_device)
    for handler in handler_collection:
        handler.remove()

    return total_ops, total_params
