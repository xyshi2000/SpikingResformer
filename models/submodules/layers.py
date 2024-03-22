import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, functional
from spikingjelly.activation_based import surrogate, neuron

from torch.nn.common_types import _size_2_t


class IF(neuron.IFNode):
    def __init__(self):
        super().__init__(v_threshold=1., v_reset=0., surrogate_function=surrogate.ATan(),
                         detach_reset=True, step_mode='m', backend='cupy', store_v_seq=False)


class LIF(neuron.LIFNode):
    def __init__(self):
        super().__init__(tau=2., decay_input=True, v_threshold=1., v_reset=0.,
                         surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m',
                         backend='cupy', store_v_seq=False)


class PLIF(neuron.ParametricLIFNode):
    def __init__(self):
        super().__init__(init_tau=2., decay_input=True, v_threshold=1., v_reset=0.,
                         surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m',
                         backend='cupy', store_v_seq=False)


class BN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True,
                                 track_running_stats=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(
                f'expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!')
        return functional.seq_to_ann_forward(x, self.bn)


class SpikingMatmul(nn.Module):
    def __init__(self, spike: str) -> None:
        super().__init__()
        assert spike == 'l' or spike == 'r' or spike == 'both'
        self.spike = spike

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        return torch.matmul(left, right)


class Conv3x3(layer.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: _size_2_t = 1,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation,
                         dilation=dilation, groups=groups, bias=bias, padding_mode='zeros',
                         step_mode='m')


class Conv1x1(layer.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: _size_2_t = 1,
        bias: bool = False,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                         dilation=1, groups=1, bias=bias, padding_mode='zeros', step_mode='m')


class Linear(layer.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, step_mode='m')
