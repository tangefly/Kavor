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


class UnquantizedMoeMethod:
    """不量化的 MoE 专家:按专家堆叠的 bf16 3D 权重。

    gate/up 合并成 ``gate_up_proj``(E, 2*local_inter, H):前一半 gate、后一半
    up;``down_proj`` 是 (E, H, local_inter)。local_inter 已按 TP 切过。
    """

    def create_weights(self, layer, num_experts, local_inter, hidden_size, weight_loader):
        gate_up = nn.Parameter(
            torch.empty(num_experts, 2 * local_inter, hidden_size))
        gate_up.weight_loader = weight_loader
        layer.gate_up_proj = gate_up
        down = nn.Parameter(torch.empty(num_experts, hidden_size, local_inter))
        down.weight_loader = weight_loader
        layer.down_proj = down
        # 统一接口:未量化时 scale 不存在,forward 里靠 None 走无 scale 的 kernel
        layer.register_parameter("gate_up_scale_inv", None)
        layer.register_parameter("down_scale_inv", None)

    def prepare_kernel(self, layer):
        pass

    def apply(self, layer, x, topk_ids, topk_weights):
        from kavor.kernels.moe import fused_moe

        return fused_moe(
            x, topk_ids, topk_weights,
            layer.gate_up_proj, None, layer.down_proj, None)
