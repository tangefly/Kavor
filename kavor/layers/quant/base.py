import torch
from torch import nn
import torch.nn.functional as F


class BlockQuantScaleParameter(nn.Parameter):
    """128x128 blockwise weight scale 的类型标记。

    Linear 的 weight_loader 靠 isinstance(param, BlockQuantScaleParameter)
    把"元素单位的分片"换算成"block 单位的分片";权重本身不走这条分支。
    """


class UnquantizedLinearMethod:
    """不量化的 Linear:保持 Kavor 原有行为(默认 dtype 权重 + F.linear)。"""

    def create_weights(self, layer, input_size, output_size, weight_loader):
        weight = nn.Parameter(torch.empty(output_size, input_size))
        weight.weight_loader = weight_loader
        layer.weight = weight

    def apply(self, layer, x, bias):
        return F.linear(x, layer.weight, bias)
