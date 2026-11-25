import torch
import triton
import triton.language as tl
from torch import nn


@triton.jit
def _sample_kernel(
    logits_ptr,
    temps_ptr,
    seed_ptr,
    out_ptr,
    vocab_size,
    BLOCK: tl.constexpr,
):
    """逐序列一格:token = argmax(logits / T + Gumbel)。

    Gumbel-max 与原先的 softmax 后除 exponential 噪声取 argmax 在分布上严格
    等价(argmax(log softmax − log Exp(1)) ≡ argmax(logit/T + Gumbel)),把
    5 个 pass 的 fp32 概率物化折叠成一遍流式扫描。temperature == 0 走 greedy。
    噪声用 philox,offset = row·V + col 保证行间独立;seed 由外部每步 +1
    (graph 捕获后由 replay 推进)。
    """
    row = tl.program_id(0)
    temp = tl.load(temps_ptr + row)
    seed = tl.load(seed_ptr)
    best = -float("inf")
    best_idx = 0
    for v0 in range(0, vocab_size, BLOCK):
        offs = v0 + tl.arange(0, BLOCK)
        mask = offs < vocab_size
        lg = tl.load(logits_ptr + row.to(tl.int64) * vocab_size + offs,
                     mask=mask, other=-float("inf")).to(tl.float32)
        if temp > 0.0:
            u = tl.rand(seed, offs + row * vocab_size)
            val = lg / temp + (-tl.log(-tl.log(u)))
        else:
            val = lg
        chunk_max = tl.max(val, axis=0)
        # tie 取最先出现(与 torch.argmax 语义一致):块内取最小下标,块间严格大于才替换
        chunk_idx = tl.min(tl.where(val == chunk_max, offs, vocab_size), axis=0)
        take = chunk_max > best
        best_idx = tl.where(take, chunk_idx, best_idx)
        best = tl.where(take, chunk_max, best)
    tl.store(out_ptr + row, best_idx.to(tl.int64))


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()
        # graph 捕获后 python 自增不会随 replay 执行,seed 必须是 GPU 上的
        # 持久 buffer,由捕获在内的 add_(1) 每步推进
        self.register_buffer("seed", torch.zeros((), dtype=torch.int32, device="cuda"))

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor,
                out: torch.Tensor | None = None):
        assert logits.is_contiguous() and temperatures.is_contiguous()
        self.seed.add_(1)
        if out is None:
            out = torch.empty(logits.shape[0], dtype=torch.int64, device=logits.device)
        vocab_size = logits.shape[-1]
        _sample_kernel[(logits.shape[0],)](
            logits, temperatures, self.seed, out, vocab_size,
            BLOCK=4096, num_warps=4)
        return out
