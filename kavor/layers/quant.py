import torch
from torch import nn
import torch.nn.functional as F

from kavor.layers import fp8_flashinfer
from kavor.layers.fp8_kernel import fp8_blockwise_linear
from kavor.layers.fp8_utils import w8a8_block_fp8_linear


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


# auto 模式下走 w8a16 的 M 上限(仅在 FlashInfer 不可用时生效)。decode 小 M 是
# 权重带宽受限:w8a16 省掉激活量化那次读写,且带 split-K 把 N-tile 铺满 SM
# (实测 M<=32 比 w8a8 快 1.2~1.7x);M 变大后是算力受限,FP8×FP8 胜出,回到 w8a8。
W8A16_MAX_M = 32


class Fp8BlockLinearMethod:
    """FP8 E4M3 权重 + blockwise scale;fp8_mode 决定激活侧走哪条计算路径。

    flashinfer:FlashInfer/DeepGEMM 的 block-scale GEMM(kavor/layers/fp8_flashinfer.py),
        激活量化在 kernel 内部完成;比下面两条 Triton 路径快 1.4~3.7x,
        要求 SM90 + fp32 scale + N/K 对齐,不满足时构建期直接报错;
    w8a8:激活动态量化(per-token per-128-group)后进 FP8×FP8 GEMM,
        kernel vendor 自 vllm(kavor/layers/fp8_utils.py);
    w8a16:激活保持 bf16,kernel 内权重转 bf16(kavor/layers/fp8_kernel.py);
    auto:能走 FlashInfer 就走(各 M 区间都更快),否则按 M 分流
        (decode 小 M 走 w8a16,prefill 大 M 走 w8a8)。
    各 mode 共用同一份权重/scale 存储,只是 forward 分流。
    """

    def __init__(self, quant_config):
        self.weight_block_size = quant_config.weight_block_size
        self.fp8_mode = quant_config.fp8_mode

    def create_weights(self, layer, input_size, output_size, weight_loader):
        block_n, block_k = self.weight_block_size
        assert output_size % block_n == 0 and input_size % block_k == 0, (
            f"FP8 权重 [{output_size}, {input_size}] 无法按 block "
            f"{self.weight_block_size} 对齐")
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
        if self.fp8_mode == "flashinfer":
            # 显式指定时在加载期就报错,不等到第一个 forward 才回退。
            # 形状无需额外检查:128x128 block 对齐已蕴含 FlashInfer 的 N%64/K%128
            assert fp8_flashinfer.flashinfer_available(), (
                "fp8_mode='flashinfer' 但 FlashInfer 不可用(未安装或非 SM90)")

    def apply(self, layer, x, bias):
        if self.fp8_mode == "flashinfer":
            assert fp8_flashinfer.supports(x, layer.weight, layer.weight_scale_inv), (
                "fp8_mode='flashinfer' 但本次调用不满足约束(bf16 输入 / fp32 scale)")
            return fp8_flashinfer.flashinfer_block_fp8_linear(
                x, layer.weight, layer.weight_scale_inv, bias)
        if self.fp8_mode == "auto":
            if fp8_flashinfer.supports(x, layer.weight, layer.weight_scale_inv):
                return fp8_flashinfer.flashinfer_block_fp8_linear(
                    x, layer.weight, layer.weight_scale_inv, bias)
            # M 在 CUDA graph 捕获时是静态的,分流结果会随每个 bs 的图固化
            M = x.numel() // x.shape[-1]
            if M <= W8A16_MAX_M:
                return fp8_blockwise_linear(
                    x, layer.weight, layer.weight_scale_inv, bias)
            return w8a8_block_fp8_linear(
                x, layer.weight, layer.weight_scale_inv,
                self.weight_block_size, bias)
        if self.fp8_mode == "w8a8":
            return w8a8_block_fp8_linear(
                x, layer.weight, layer.weight_scale_inv,
                self.weight_block_size, bias)
        return fp8_blockwise_linear(x, layer.weight, layer.weight_scale_inv, bias)


class Fp8Config:
    """HF quantization_config 的 FP8 解析;按模块路径(prefix)决定是否量化。"""

    def __init__(self, weight_block_size, modules_to_not_convert,
                 fp8_mode="w8a8", activation_scheme=None):
        self.weight_block_size = tuple(weight_block_size)
        self.modules_to_not_convert = list(modules_to_not_convert or [])
        assert fp8_mode in ("w8a8", "w8a16", "auto", "flashinfer"), (
            f"fp8_mode 只支持 'w8a8' / 'w8a16' / 'auto' / 'flashinfer',遇到 {fp8_mode!r}")
        self.fp8_mode = fp8_mode
        # checkpoint 侧声明的激活量化方案(dynamic);实际走哪条路由由
        # fp8_mode 决定,这里只留作记录/日志
        self.activation_scheme = activation_scheme

    @classmethod
    def from_config(cls, quantization_config: dict | None, fp8_mode: str = "w8a8"):
        if not quantization_config:
            return None
        method = quantization_config.get("quant_method")
        assert method == "fp8", f"只实现了 fp8 量化,遇到 {method!r}"
        assert quantization_config.get("fmt") == "e4m3", "只支持 e4m3"
        block_size = tuple(quantization_config["weight_block_size"])
        assert block_size == (128, 128), f"只支持 128x128 block,遇到 {block_size}"
        return cls(block_size, quantization_config.get("modules_to_not_convert"),
                   fp8_mode, quantization_config.get("activation_scheme"))

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
        return Fp8BlockLinearMethod(self)
