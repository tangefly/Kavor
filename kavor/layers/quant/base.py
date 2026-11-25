import torch
import torch.nn.functional as F
from torch import nn


class BlockQuantScaleParameter(nn.Parameter):
    """Marker type for 128x128 blockwise weight scales.

    ``weight_loader`` uses ``isinstance`` to tell scale params from weights and
    convert an element-sized shard into the corresponding block-sized shard.
    """


class UnquantizedLinearMethod:
    """Unquantized linear: weights in their default dtype, computed with ``F.linear``."""

    def create_weights(self, layer, input_size, output_size, weight_loader):
        weight = nn.Parameter(torch.empty(output_size, input_size))
        weight.weight_loader = weight_loader
        layer.weight = weight

    def apply(self, layer, x, bias):
        return F.linear(x, layer.weight, bias)


class UnquantizedMoeMethod:
    """Unquantized MoE experts, stored as per-expert stacked 3D bf16 weights.

    ``gate_up_proj`` is ``(E, 2 * local_inter, H)`` with gate first and up
    second; ``down_proj`` is ``(E, H, local_inter)``. ``local_inter`` is
    already sharded by TP.
    """

    def create_weights(self, layer, num_experts, local_inter, hidden_size, weight_loader):
        gate_up = nn.Parameter(
            torch.empty(num_experts, 2 * local_inter, hidden_size))
        gate_up.weight_loader = weight_loader
        layer.gate_up_proj = gate_up
        down = nn.Parameter(torch.empty(num_experts, hidden_size, local_inter))
        down.weight_loader = weight_loader
        layer.down_proj = down
        # Same interface as quantized methods: no scales here, so the kernel is
        # selected by passing None below.
        layer.register_parameter("gate_up_scale_inv", None)
        layer.register_parameter("down_scale_inv", None)

    def prepare_kernel(self, layer):
        pass

    def apply(self, layer, x, topk_ids, topk_weights):
        from kavor.kernels.moe import fused_moe

        return fused_moe(
            x, topk_ids, topk_weights,
            layer.gate_up_proj, None, layer.down_proj, None)
