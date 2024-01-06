from torch import nn

def select_activation_function(activation):
    if isinstance(activation, str):
        if activation.lower() == 'relu':
            return nn.ReLU
        elif activation.lower() == 'softplus':
            return nn.Softplus
        else:
            raise ValueError(f"Unknown activation type {activation}")
    elif isinstance(activation, nn.Module):
        return activation
    else:
        raise ValueError(f"Unknown activation type {activation}")

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, dw_kernel, dw_padding, dw_stride=1,
                 activation=None, use_bias=False, norm_layer=None):
        super(SeparableConv2d, self).__init__()
        _activation = select_activation_function(activation)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=dw_kernel, stride=dw_stride,
                      padding=dw_padding, bias=use_bias, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
            _activation()
        )

    def forward(self, x):
        return self.body(x)