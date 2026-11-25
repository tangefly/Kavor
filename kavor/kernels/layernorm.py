import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    out_ptr,
    w_ptr,
    n_cols,
    stride_x_t,
    stride_x_h,
    stride_o_t,
    stride_o_h,
    eps,
    BLOCK: tl.constexpr,
):
    """out = orig_dtype(orig_dtype(x * rsqrt(mean(x²)+eps)) * w),逐 (token, head) 一格。

    与原 torch 实现保持同一数值路径:fp32 归一化、先转回原 dtype 再乘 weight
    (复刻 `x.to(orig).mul_(weight)` 的二次舍入)。2D 输入(heads=1)与 3D
    输入(q/k norm)共用;x/out 按各自真实 stride 寻址(qkv 的 split 视图
    不连续)。不做 fp32 中间量物化。
    """
    t = tl.program_id(0)
    h = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + t * stride_x_t + h * stride_x_h + cols,
                mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = tl.math.rsqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x * rstd).to(x_ptr.dtype.element_ty).to(tl.float32) * w
    tl.store(out_ptr + t * stride_o_t + h * stride_o_h + cols,
             y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rmsnorm_gemma_kernel(
    x_ptr,
    out_ptr,
    w_ptr,
    n_cols,
    stride_x_t,
    stride_x_h,
    stride_o_t,
    stride_o_h,
    eps,
    BLOCK: tl.constexpr,
):
    """Gemma 式 RMSNorm(Qwen3_5 全家):out = orig_dtype(fp32_norm(x) * (1 + w))。

    与 Qwen3_5RMSNorm eager 同数值路径:全程 fp32、(1+w) 在 fp32 里加、只在
    最后 cast 一次(不折叠加 1 进 bf16 权重,那会损失 ~2^-9 相对精度)。
    """
    t = tl.program_id(0)
    h = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + t * stride_x_t + h * stride_x_h + cols,
                mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = tl.math.rsqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * (1.0 + w)
    tl.store(out_ptr + t * stride_o_t + h * stride_o_h + cols,
             y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rmsnorm_gated_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    w_ptr,
    n_cols,
    stride_x_t,
    stride_x_h,
    stride_g_t,
    stride_g_h,
    stride_o_t,
    stride_o_h,
    eps,
    BLOCK: tl.constexpr,
):
    """Qwen3_5RMSNormGated: out = orig_dtype( orig(norm(x)) · w) · silu_f32(gate) )。

    复刻 eager 的混合精度路径:norm 在 fp32 → cast 回原 dtype;与 w 的乘积
    先舍入回原 dtype(torch 对 bf16 逐元素乘以 fp32 opmath 计算后落回 bf16),
    再乘 fp32 silu(gate),最后 cast 一次。逐 (token, head) 一格。
    """
    t = tl.program_id(0)
    h = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    p = t * stride_x_t + h * stride_x_h + cols
    x = tl.load(x_ptr + p, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = tl.math.rsqrt(var + eps)
    xn = (x * rstd).to(x_ptr.dtype.element_ty)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(x_ptr.dtype.element_ty).to(tl.float32)
    g = tl.load(gate_ptr + t * stride_g_t + h * stride_g_h + cols,
                mask=mask, other=0.0).to(tl.float32)
    g = g * tl.sigmoid(g)
    tl.store(out_ptr + t * stride_o_t + h * stride_o_h + cols,
             (y * g).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _add_rmsnorm_kernel(
    x_ptr,
    res_ptr,
    out_ptr,
    new_res_ptr,
    w_ptr,
    n_cols,
    eps,
    BLOCK: tl.constexpr,
):
    """new_residual = orig_dtype(x + residual);out = orig_dtype(orig_dtype(norm(new_residual)) * w)。

    与原 add_rms_forward 同数值路径:加法在 fp32 做,残差以原 dtype 落盘。
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(res_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    x = x + res
    tl.store(new_res_ptr + row * n_cols + cols, x.to(new_res_ptr.dtype.element_ty), mask=mask)
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = tl.math.rsqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x * rstd).to(x_ptr.dtype.element_ty).to(tl.float32) * w
    tl.store(out_ptr + row * n_cols + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


# 单 kernel 单 block 能装下的最大列数;更大的 hidden(当前模型族没有)退回 eager
_MAX_FUSED_COLS = 8192


def _num_warps(block: int) -> int:
    return max(1, min(8, block // 256))


def rmsnorm_gemma(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Gemma 式 (1+w) RMSNorm,fp32 全程、末尾一次 cast,数值同 Qwen3_5RMSNorm eager。

    n_cols 超出单 block 上限(当前模型族不会)或最后一维不连续时退回 eager。
    """
    n_cols = x.shape[-1]
    if n_cols > _MAX_FUSED_COLS or x.stride(-1) != 1:
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        return (xf * torch.rsqrt(var + eps) * (1.0 + weight.float())).type_as(x)
    if x.dim() == 3:
        tokens, heads = x.shape[0], x.shape[1]
        sx_t, sx_h = x.stride(0), x.stride(1)
    else:
        tokens, heads = x.shape[0], 1
        sx_t, sx_h = x.stride(0), 0
    out = torch.empty_like(x) if x.is_contiguous() else torch.empty(
        x.shape, dtype=x.dtype, device=x.device)
    BLOCK = max(triton.next_power_of_2(n_cols), 16)
    _rmsnorm_gemma_kernel[(tokens, heads)](
        x, out, weight, n_cols, sx_t, sx_h, out.stride(0), out.stride(-2),
        eps, BLOCK=BLOCK, num_warps=_num_warps(BLOCK))
    return out


class RMSNormGated(nn.Module):
    """norm + silu(gate) 单 kernel(Qwen3_5 线性注意力的 output norm)。

    数值路径复刻 Qwen3_5RMSNormGated eager(见 _rmsnorm_gated_kernel docstring)。
    x/gate 形状相同,支持 [T, D] 与 [T, H, D]。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        n_cols = x.shape[-1]
        if x.dim() != gate.dim():
            x = x.reshape(*gate.shape)
        if (n_cols > _MAX_FUSED_COLS or x.stride(-1) != 1 or gate.stride(-1) != 1
                or x.shape != gate.shape):
            input_dtype = x.dtype
            h = x.float()
            var = h.pow(2).mean(dim=-1, keepdim=True)
            h = h * torch.rsqrt(var + self.eps)
            h = self.weight * h.to(input_dtype)
            return (h * F.silu(gate.float())).to(input_dtype)
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        if x.dim() == 3:
            tokens, heads = x.shape[0], x.shape[1]
            sg_t, sg_h = gate.stride(0), gate.stride(1)
        else:
            tokens, heads = x.shape[0], 1
            sg_t, sg_h = gate.stride(0), 0
        sx_t, sx_h = (x.stride(0), x.stride(1)) if x.dim() == 3 else (x.stride(0), 0)
        BLOCK = max(triton.next_power_of_2(n_cols), 16)
        _rmsnorm_gated_kernel[(tokens, heads)](
            x, gate, out, self.weight, n_cols,
            sx_t, sx_h, sg_t, sg_h, out.stride(0), out.stride(-2),
            self.eps, BLOCK=BLOCK, num_warps=_num_warps(BLOCK))
        return out


class RMSNorm(nn.Module):
    """fused RMSNorm / fused add+RMSNorm(Triton 实现)。

    替换原先的 @torch.compile 版本:那份实现被 2D(hidden)与 3D(q/k norm)
    输入来回复用,dynamo 变体 churn 触发 recompile_limit 后整体回退 eager,
    每层多出 ~8 个微型 elementwise kernel(见 decode profile)。add 路径
    原地写回残差(各调用点的输入随后即弃),rms 路径写新张量(第一层的
    `h, res = norm(h), h` 依赖返回值是新 tensor,不能原地)。
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        n_cols = x.shape[-1]
        if x.dim() == 3:
            tokens, heads = x.shape[0], x.shape[1]
            sx_t, sx_h = x.stride(0), x.stride(1)
        else:
            tokens, heads = x.shape[0], 1
            sx_t, sx_h = x.stride(0), 0
        if n_cols > _MAX_FUSED_COLS or x.stride(-1) != 1:
            orig_dtype = x.dtype
            xf = x.float()
            var = xf.pow(2).mean(dim=-1, keepdim=True)
            xf.mul_(torch.rsqrt(var + self.eps))
            return xf.to(orig_dtype).mul_(self.weight)
        out = torch.empty_like(x) if x.is_contiguous() else torch.empty(x.shape, dtype=x.dtype, device=x.device)
        BLOCK = max(triton.next_power_of_2(n_cols), 16)
        _rmsnorm_kernel[(tokens, heads)](
            x, out, self.weight, n_cols, sx_t, sx_h,
            out.stride(0), out.stride(-2), self.eps,
            BLOCK=BLOCK, num_warps=_num_warps(BLOCK))
        return out

    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """数值参考(融合版等价于):
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual
        """
        n_cols = x.shape[-1]
        if (n_cols <= _MAX_FUSED_COLS and x.dim() == 2
                and x.is_contiguous() and residual.is_contiguous()):
            out = torch.empty_like(x)
            BLOCK = max(triton.next_power_of_2(n_cols), 16)
            _add_rmsnorm_kernel[(x.shape[0],)](
                x, residual, out, residual, self.weight, n_cols, self.eps,
                BLOCK=BLOCK, num_warps=_num_warps(BLOCK))
            return out, residual
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        return x.to(orig_dtype).mul_(self.weight), residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
