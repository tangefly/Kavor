import torch
from torch import nn

from kavor.kernels.fp8.registry import Fp8LinearConfig, select_kernel
from kavor.layers.quant.base import (
    BlockQuantScaleParameter,
    UnquantizedLinearMethod,
    UnquantizedMoeMethod,
)


class Fp8BlockLinearMethod:
    """FP8 E4M3 weights with 128x128 blockwise scales.

    The backend is chosen once in ``prepare_kernel``, so forward is CUDA-graph safe.
    """

    def __init__(self, quant_config, prefix: str = ""):
        self.weight_block_size = quant_config.weight_block_size
        self.fp8_backend = quant_config.fp8_backend
        self.prefix = prefix
        self.kernel = None

    def create_weights(self, layer, input_size, output_size, weight_loader):
        block_n, block_k = self.weight_block_size
        assert output_size % block_n == 0 and input_size % block_k == 0, (
            f"FP8 weight [{output_size}, {input_size}] is not aligned to block "
            f"{self.weight_block_size}"
        )
        # Marlin needs the model dtype for its scale exponent bias and activation dtype.
        layer.orig_dtype = torch.get_default_dtype()
        
        weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn),
            requires_grad=False
        )
        weight.weight_loader = weight_loader
        layer.weight = weight
        layer.weight_block_size = self.weight_block_size
        
        # fp32 scales: one per block, and FlashInfer requires fp32.
        scale = BlockQuantScaleParameter(
            torch.empty(output_size // block_n, input_size // block_k, dtype=torch.float32),
            requires_grad=False)
        scale.weight_loader = weight_loader
        layer.register_parameter("weight_scale_inv", scale)

    def prepare_kernel(self, layer):
        """Pick a kernel and convert the layout once, after loading."""
        cfg = Fp8LinearConfig(
            n=layer.weight.shape[0],
            k=layer.weight.shape[1],
            block_n=self.weight_block_size[0],
            block_k=self.weight_block_size[1],
            input_dtype=getattr(layer, "orig_dtype", torch.get_default_dtype()),
            has_bias=getattr(layer, "bias", None) is not None,
        )
        self.kernel = select_kernel(cfg, self.fp8_backend, prefix=self.prefix)
        self.kernel.prepare(layer)

    def apply(self, layer, x, bias):
        # Lazy prepare for paths that skip loading-time setup.
        if self.kernel is None:
            self.prepare_kernel(layer)
        return self.kernel.apply(layer, x, bias)


class Fp8BlockMoeMethod:
    """FP8 E4M3 expert weights with 128x128 blockwise scales, stacked per expert.

    ``weight_loader`` packs per-expert 2D checkpoint weights into
    ``gate_up_proj`` (E, 2*local_inter, H) and ``down_proj`` (E, H, local_inter).
    """

    def __init__(self, quant_config, prefix: str = ""):
        self.weight_block_size = quant_config.weight_block_size
        self.prefix = prefix

    def create_weights(self, layer, num_experts, local_inter, hidden_size, weight_loader):
        block_n, block_k = self.weight_block_size
        two_inter = 2 * local_inter
        assert two_inter % block_n == 0 and hidden_size % block_k == 0, (
            f"Expert gate/up weight [{two_inter}, {hidden_size}] is not aligned "
            f"to block {self.weight_block_size}"
        )
        assert hidden_size % block_n == 0 and local_inter % block_k == 0, (
            f"Expert down weight [{hidden_size}, {local_inter}] is not aligned "
            f"to block {self.weight_block_size}"
        )
        layer.orig_dtype = torch.get_default_dtype()
        layer.weight_block_size = self.weight_block_size

        gate_up = nn.Parameter(
            torch.empty(num_experts, two_inter, hidden_size, dtype=torch.float8_e4m3fn), 
            requires_grad=False
        )
        gate_up.weight_loader = weight_loader
        layer.gate_up_proj = gate_up
        
        gate_up_scale = BlockQuantScaleParameter(
            torch.empty(num_experts, two_inter // block_n, hidden_size // block_k,
            dtype=torch.float32),
            requires_grad=False
        )
        gate_up_scale.weight_loader = weight_loader
        layer.gate_up_scale_inv = gate_up_scale

        down = nn.Parameter(
            torch.empty(num_experts, hidden_size, local_inter,
                        dtype=torch.float8_e4m3fn),
            requires_grad=False)
        down.weight_loader = weight_loader
        layer.down_proj = down
        
        down_scale = BlockQuantScaleParameter(
            torch.empty(num_experts, hidden_size // block_n, local_inter // block_k,
                        dtype=torch.float32),
            requires_grad=False)
        down_scale.weight_loader = weight_loader
        layer.down_scale_inv = down_scale

    def prepare_kernel(self, layer):
        from kavor.utils.logger import logger

        logger.debug(
            "%sFP8 MoE experts: E=%d gate_up=%s down=%s",
            f"{self.prefix}: " if self.prefix else "",
            layer.gate_up_proj.shape[0],
            tuple(layer.gate_up_proj.shape), tuple(layer.down_proj.shape)
        )

    def apply(self, layer, x, topk_ids, topk_weights):
        from kavor.kernels.moe import fused_moe

        return fused_moe(
            x, topk_ids, topk_weights,
            layer.gate_up_proj, layer.gate_up_scale_inv,
            layer.down_proj, layer.down_scale_inv)


class Fp8Config:
    """Parsed HF FP8 config; decides per module prefix whether to quantize."""

    def __init__(
        self, 
        weight_block_size, 
        modules_to_not_convert,
        fp8_backend="auto", 
        activation_scheme=None
    ):
        from kavor.kernels.fp8.registry import BACKENDS

        self.weight_block_size = tuple(weight_block_size)
        self.modules_to_not_convert = list(modules_to_not_convert or [])
        assert fp8_backend in BACKENDS, (
            f"fp8_backend must be one of {', '.join(BACKENDS)}, got {fp8_backend!r}")
        self.fp8_backend = fp8_backend
        # Only recorded for logs; ``fp8_backend`` picks the actual route.
        self.activation_scheme = activation_scheme

    @classmethod
    def from_config(cls, quantization_config: dict | None, fp8_backend: str = "auto"):
        if not quantization_config:
            return None
        method = quantization_config.get("quant_method")
        assert method == "fp8", f"only fp8 quantization is implemented, got {method!r}"
        assert quantization_config.get("fmt") == "e4m3", "only e4m3 is supported"
        block_size = tuple(quantization_config["weight_block_size"])
        assert block_size == (128, 128), f"only 128x128 blocks are supported, got {block_size}"
        return cls(block_size, quantization_config.get("modules_to_not_convert"),
                   fp8_backend, quantization_config.get("activation_scheme"))

    def is_layer_skipped(self, prefix: str) -> bool:
        """Matches full paths (model.layers.0.xxx) or leaf names (xxx); unmatched layers are quantized."""
        return any(
            prefix == name or prefix.endswith(f".{name}")
            for name in self.modules_to_not_convert
        )

    def get_linear_method(self, prefix: str):
        if self.is_layer_skipped(prefix):
            return UnquantizedLinearMethod()
        return Fp8BlockLinearMethod(self, prefix=prefix)

    def get_moe_method(self, prefix: str):
        if self.is_layer_skipped(prefix):
            return UnquantizedMoeMethod()
        return Fp8BlockMoeMethod(self, prefix=prefix)
