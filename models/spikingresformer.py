import math
import torch
import torch.nn as nn
from .submodules.layers import Conv3x3, Conv1x1, LIF, PLIF, BN, Linear, SpikingMatmul
from spikingjelly.activation_based import layer
from typing import Any, List, Mapping
from timm.models.registry import register_model


class GWFFN(nn.Module):
    def __init__(self, in_channels, num_conv=1, ratio=4, group_size=64, activation=LIF):
        super().__init__()
        inner_channels = in_channels * ratio
        self.up = nn.Sequential(
            activation(),
            Conv1x1(in_channels, inner_channels),
            BN(inner_channels),
        )
        self.conv = nn.ModuleList()
        for _ in range(num_conv):
            self.conv.append(
                nn.Sequential(
                    activation(),
                    Conv3x3(inner_channels, inner_channels, groups=inner_channels // group_size),
                    BN(inner_channels),
                ))
        self.down = nn.Sequential(
            activation(),
            Conv1x1(inner_channels, in_channels),
            BN(in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_feat_out = x.clone()
        x = self.up(x)
        x_feat_in = x.clone()
        for m in self.conv:
            x = m(x)
        x = x + x_feat_in
        x = self.down(x)
        x = x + x_feat_out
        return x


class DSSA(nn.Module):
    def __init__(self, dim, num_heads, lenth, patch_size, activation=LIF):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.lenth = lenth
        self.register_buffer('firing_rate_x', torch.zeros(1, 1, num_heads, 1, 1))
        self.register_buffer('firing_rate_attn', torch.zeros(1, 1, num_heads, 1, 1))
        self.init_firing_rate_x = False
        self.init_firing_rate_attn = False
        self.momentum = 0.999

        self.activation_in = activation()

        self.W = layer.Conv2d(dim, dim * 2, patch_size, patch_size, bias=False, step_mode='m')
        self.norm = BN(dim * 2)
        self.matmul1 = SpikingMatmul('r')
        self.matmul2 = SpikingMatmul('r')
        self.activation_attn = activation()
        self.activation_out = activation()

        self.Wproj = Conv1x1(dim, dim)
        self.norm_proj = BN(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # X: [T, B, C, H, W]
        T, B, C, H, W = x.shape
        x_feat = x.clone()
        x = self.activation_in(x)

        y = self.W(x)
        y = self.norm(y)
        y = y.reshape(T, B, self.num_heads, 2 * C // self.num_heads, -1)
        y1, y2 = y[:, :, :, :C // self.num_heads, :], y[:, :, :, C // self.num_heads:, :]
        x = x.reshape(T, B, self.num_heads, C // self.num_heads, -1)

        if self.training:
            firing_rate_x = x.detach().mean((0, 1, 3, 4), keepdim=True)
            if not self.init_firing_rate_x and torch.all(self.firing_rate_x == 0):
                self.firing_rate_x = firing_rate_x
            self.init_firing_rate_x = True
            self.firing_rate_x = self.firing_rate_x * self.momentum + firing_rate_x * (
                1 - self.momentum)
        scale1 = 1. / torch.sqrt(self.firing_rate_x * (self.dim // self.num_heads))
        attn = self.matmul1(y1.transpose(-1, -2), x)
        attn = attn * scale1
        attn = self.activation_attn(attn)

        if self.training:
            firing_rate_attn = attn.detach().mean((0, 1, 3, 4), keepdim=True)
            if not self.init_firing_rate_attn and torch.all(self.firing_rate_attn == 0):
                self.firing_rate_attn = firing_rate_attn
            self.init_firing_rate_attn = True
            self.firing_rate_attn = self.firing_rate_attn * self.momentum + firing_rate_attn * (
                1 - self.momentum)
        scale2 = 1. / torch.sqrt(self.firing_rate_attn * self.lenth)
        out = self.matmul2(y2, attn)
        out = out * scale2
        out = out.reshape(T, B, C, H, W)
        out = self.activation_out(out)

        out = self.Wproj(out)
        out = self.norm_proj(out)
        out = out + x_feat
        return out


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, activation=LIF) -> None:
        super().__init__()
        self.conv = Conv3x3(in_channels, out_channels, stride=stride)
        self.norm = BN(out_channels)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(x)
        x = self.conv(x)
        x = self.norm(x)
        return x


class SpikingResformer(nn.Module):
    def __init__(
        self,
        layers: List[List[str]],
        planes: List[int],
        num_heads: List[int],
        patch_sizes: List[int],
        img_size=224,
        T=4,
        in_channels=3,
        num_classes=1000,
        prologue=None,
        group_size=64,
        activation=LIF,
        **kwargs,
    ):
        super().__init__()
        self.T = T
        self.skip = ['prologue.0', 'classifier']
        assert len(planes) == len(layers) == len(num_heads) == len(patch_sizes)

        if prologue is None:
            self.prologue = nn.Sequential(
                layer.Conv2d(in_channels, planes[0], 7, 2, 3, bias=False, step_mode='m'),
                BN(planes[0]),
                layer.MaxPool2d(kernel_size=3, stride=2, padding=1, step_mode='m'),
            )
            img_size = img_size // 4
        else:
            self.prologue = prologue

        self.layers = nn.Sequential()
        for idx in range(len(planes)):
            sub_layers = nn.Sequential()
            if idx != 0:
                sub_layers.append(
                    DownsampleLayer(planes[idx - 1], planes[idx], stride=2, activation=activation))
                img_size = img_size // 2
            for name in layers[idx]:
                if name == 'DSSA':
                    sub_layers.append(
                        DSSA(planes[idx], num_heads[idx], (img_size // patch_sizes[idx])**2,
                             patch_sizes[idx], activation=activation))
                elif name == 'GWFFN':
                    sub_layers.append(
                        GWFFN(planes[idx], group_size=group_size, activation=activation))
                else:
                    raise ValueError(name)
            self.layers.append(sub_layers)

        self.avgpool = layer.AdaptiveAvgPool2d((1, 1), step_mode='m')
        self.classifier = Linear(planes[-1], num_classes)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def transfer(self, state_dict: Mapping[str, Any]):
        _state_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}
        return self.load_state_dict(_state_dict, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            assert x.dim() == 5
        else:
            #### [B, T, C, H, W] -> [T, B, C, H, W]
            x = x.transpose(0, 1)
        x = self.prologue(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x

    def no_weight_decay(self):
        ret = set()
        for name, module in self.named_modules():
            if isinstance(module, PLIF):
                ret.add(name + '.w')
        return ret


@register_model
def spikingresformer_ti(**kwargs):
    return SpikingResformer(
        [
            ['DSSA', 'GWFFN'] * 1,
            ['DSSA', 'GWFFN'] * 2,
            ['DSSA', 'GWFFN'] * 3, ],
        [64, 192, 384],
        [1, 3, 6],
        [4, 2, 1],
        in_channels=3,
        **kwargs,
    )


@register_model
def spikingresformer_s(**kwargs):
    return SpikingResformer(
        [
            ['DSSA', 'GWFFN'] * 1,
            ['DSSA', 'GWFFN'] * 2,
            ['DSSA', 'GWFFN'] * 3, ],
        [64, 256, 512],
        [1, 4, 8],
        [4, 2, 1],
        in_channels=3,
        **kwargs,
    )


@register_model
def spikingresformer_m(**kwargs):
    return SpikingResformer(
        [
            ['DSSA', 'GWFFN'] * 1,
            ['DSSA', 'GWFFN'] * 2,
            ['DSSA', 'GWFFN'] * 3, ],
        [64, 384, 768],
        [1, 6, 12],
        [4, 2, 1],
        in_channels=3,
        **kwargs,
    )


@register_model
def spikingresformer_l(**kwargs):
    return SpikingResformer(
        [
            ['DSSA', 'GWFFN'] * 1,
            ['DSSA', 'GWFFN'] * 2,
            ['DSSA', 'GWFFN'] * 3, ],
        [128, 512, 1024],
        [2, 8, 16],
        [4, 2, 1],
        in_channels=3,
        **kwargs,
    )


@register_model
def spikingresformer_dvsg(**kwargs):
    return SpikingResformer(
        [
            ['DSSA', 'GWFFN'] * 1,
            ['DSSA', 'GWFFN'] * 2,
            ['DSSA', 'GWFFN'] * 3, ],
        [32, 96, 192],
        [1, 3, 6],
        [4, 2, 1],
        in_channels=3,
        prologue=nn.Sequential(
            layer.Conv2d(3, 32, 3, 1, 1, bias=False, step_mode='m'),
            BN(32),
        ),
        group_size=32,
        activation=PLIF,
        **kwargs,
    )


@register_model
def spikingresformer_cifar(**kwargs):
    return SpikingResformer(
        [
            ['DSSA', 'GWFFN'] * 1,
            ['DSSA', 'GWFFN'] * 2,
            ['DSSA', 'GWFFN'] * 3, ],
        [64, 192, 384],
        [1, 3, 6],
        [4, 2, 1],
        in_channels=3,
        prologue=nn.Sequential(
            layer.Conv2d(3, 64, 3, 1, 1, bias=False, step_mode='m'),
            BN(64),
        ),
        **kwargs,
    )
