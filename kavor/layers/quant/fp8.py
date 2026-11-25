import torch
from torch import nn

from kavor.kernels.fp8.registry import Fp8LinearConfig, select_kernel
from kavor.layers.quant.base import (
    BlockQuantScaleParameter,
    UnquantizedLinearMethod,
    UnquantizedMoeMethod,
)


class Fp8BlockLinearMethod:
    """FP8 E4M3 权重 + 128×128 blockwise scale。

    具体激活侧走哪条 GEMM 由 ``fp8_backend`` 决定:
    ``auto``/``flashinfer``/``marlin``/``triton_w8a16``/``triton_w8a8``。
    选择发生在 ``prepare_kernel``(加载期,每个 layer 固定),forward 只调用
    选中的 kernel,不再按 M 分流,因此 CUDA graph 是安全的。

    所有后端共用同一份 checkpoint 形态(weight ``[N,K]`` fp8、
    weight_scale_inv ``[N/128,K/128]`` fp32);Marlin 会在 prepare 阶段把它们
    原地替换成 repack 后的 int32 权重与 permute 后的 scale。
    """

    def __init__(self, quant_config, prefix: str = ""):
        self.weight_block_size = quant_config.weight_block_size
        self.fp8_backend = quant_config.fp8_backend
        self.prefix = prefix
        self.kernel = None

    def create_weights(self, layer, input_size, output_size, weight_loader):
        block_n, block_k = self.weight_block_size
        assert output_size % block_n == 0 and input_size % block_k == 0, (
            f"FP8 权重 [{output_size}, {input_size}] 无法按 block "
            f"{self.weight_block_size} 对齐")
        # 模型 dtype 由 model_runner 在构造模型前设置;Marlin 需要它来选
        # scale 的指数偏移(bf16 2^120 / fp16 2^8)和激活类型。
        layer.orig_dtype = torch.get_default_dtype()
        weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn),
            requires_grad=False)
        weight.weight_loader = weight_loader
        layer.weight = weight
        # weight_loader 的 scale 分支需要 block 大小来做单位换算
        layer.weight_block_size = self.weight_block_size
        # scale 用 fp32:块内一个标量,精度不该被模型 dtype 拖累;
        # FlashInfer 也要求 fp32
        scale = BlockQuantScaleParameter(
            torch.empty(output_size // block_n, input_size // block_k,
                        dtype=torch.float32),
            requires_grad=False)
        scale.weight_loader = weight_loader
        layer.register_parameter("weight_scale_inv", scale)

    def prepare_kernel(self, layer):
        """在权重加载完成后调用一次:选 kernel 并做一次性布局转换。"""
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
        # 测试/非常规加载路径可能没走 prepare_kernel;这里懒触发一次。
        if self.kernel is None:
            self.prepare_kernel(layer)
        return self.kernel.apply(layer, x, bias)


class Fp8BlockMoeMethod:
    """FP8 E4M3 专家权重 + 128×128 blockwise scale(按专家堆叠)。

    checkpoint 里专家是每个专家一组 2D 权重(``experts.<i>.gate_proj`` 等),
    加载期由 ``Qwen3_5MoeExperts.weight_loader`` 拼成 3D 参数:

    - ``gate_up_proj``  (E, 2*local_inter, H)  fp8,gate 在前、up 在后
    - ``gate_up_scale_inv`` (E, 2*local_inter/128, H/128) fp32
    - ``down_proj`` (E, H, local_inter) fp8
    - ``down_scale_inv`` (E, H/128, local_inter/128) fp32

    具体 GEMM 由 ``kavor/kernels/moe.py`` 的融合 kernel 做(W8A16,块 scale
    乘在部分和上,数值路径同 triton_w8a16),不需要像 Linear 那样在多个
    后端之间选。
    """

    def __init__(self, quant_config, prefix: str = ""):
        self.weight_block_size = quant_config.weight_block_size
        self.prefix = prefix

    def create_weights(self, layer, num_experts, local_inter, hidden_size, weight_loader):
        block_n, block_k = self.weight_block_size
        two_inter = 2 * local_inter
        assert two_inter % block_n == 0 and hidden_size % block_k == 0, (
            f"专家 gate/up 权重 [{two_inter}, {hidden_size}] 无法按 block "
            f"{self.weight_block_size} 对齐")
        assert hidden_size % block_n == 0 and local_inter % block_k == 0, (
            f"专家 down 权重 [{hidden_size}, {local_inter}] 无法按 block "
            f"{self.weight_block_size} 对齐")
        layer.orig_dtype = torch.get_default_dtype()
        layer.weight_block_size = self.weight_block_size

        gate_up = nn.Parameter(
            torch.empty(num_experts, two_inter, hidden_size,
                        dtype=torch.float8_e4m3fn),
            requires_grad=False)
        gate_up.weight_loader = weight_loader
        layer.gate_up_proj = gate_up
        gate_up_scale = BlockQuantScaleParameter(
            torch.empty(num_experts, two_inter // block_n, hidden_size // block_k,
                        dtype=torch.float32),
            requires_grad=False)
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
            tuple(layer.gate_up_proj.shape), tuple(layer.down_proj.shape))

    def apply(self, layer, x, topk_ids, topk_weights):
        from kavor.kernels.moe import fused_moe

        return fused_moe(
            x, topk_ids, topk_weights,
            layer.gate_up_proj, layer.gate_up_scale_inv,
            layer.down_proj, layer.down_scale_inv)


class Fp8Config:
    """HF quantization_config 的 FP8 解析;按模块路径(prefix)决定是否量化。"""

    def __init__(self, weight_block_size, modules_to_not_convert,
                 fp8_backend="auto", activation_scheme=None):
        from kavor.kernels.fp8.registry import BACKENDS

        self.weight_block_size = tuple(weight_block_size)
        self.modules_to_not_convert = list(modules_to_not_convert or [])
        assert fp8_backend in BACKENDS, (
            f"fp8_backend 只支持 {', '.join(BACKENDS)},遇到 {fp8_backend!r}")
        self.fp8_backend = fp8_backend
        # checkpoint 侧声明的激活量化方案(dynamic);实际走哪条路由由
        # fp8_backend 决定,这里只留作记录/日志
        self.activation_scheme = activation_scheme

    @classmethod
    def from_config(cls, quantization_config: dict | None, fp8_backend: str = "auto"):
        if not quantization_config:
            return None
        method = quantization_config.get("quant_method")
        assert method == "fp8", f"只实现了 fp8 量化,遇到 {method!r}"
        assert quantization_config.get("fmt") == "e4m3", "只支持 e4m3"
        block_size = tuple(quantization_config["weight_block_size"])
        assert block_size == (128, 128), f"只支持 128x128 block,遇到 {block_size}"
        return cls(block_size, quantization_config.get("modules_to_not_convert"),
                   fp8_backend, quantization_config.get("activation_scheme"))

    def is_layer_skipped(self, prefix: str) -> bool:
        """modules_to_not_convert 里同时存在完整路径(model.layers.0.xxx)
        和末段名(xxx)两种写法,都支持;没命中的默认量化,配错了由加载校验兜底。"""
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
