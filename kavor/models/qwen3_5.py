import itertools
from collections.abc import Callable
import torch.nn.functional as F

import torch
from torch import nn
import torch.distributed as dist

from kavor.layers.activation import SiluAndMul
from kavor.layers.attention import Attention
from kavor.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from kavor.layers.layernorm import RMSNorm
from kavor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from kavor.layers.rotary_embedding import apply_rotary_emb
from kavor.utils.context import get_context
from kavor.utils.mm_plan import plan_shards

from transformers.activations import ACT2FN
from transformers.utils import (
    is_torchdynamo_exporting,
    torch_compilable_check,
)
from transformers.utils.generic import (
    get_max_seqlen,
    is_flash_attention_requested,
    maybe_autocast,
)
from transformers.vision_utils import (
    get_vision_attention_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)
from transformers import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig


class Qwen3_5VisionRotaryEmbedding(nn.Module):

    def __init__(self, config: Qwen3_5VisionConfig, device=None):
        super().__init__()
        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_axial_rope_parameters
        if self.rope_type != "axial":
            raise ValueError(f"{self.__class__.__name__} supports only axial rope, but requested {self.rope_type}")
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.inv_freq = nn.Buffer(inv_freq, persistent=False)
        self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_axial_rope_parameters(
        config: Qwen3_5VisionConfig, device=None, **kwargs
    ) -> tuple[torch.Tensor, float]:
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        spatial_dim = dim // 2

        attention_factor = 1.0  # Unused in this type of RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim))
        return inv_freq.to(device), attention_factor

    @torch.no_grad()
    def forward(self, x, position_ids):
        # position_ids: (2, N) — row 0 = h coords, row 1 = w coords
        position_ids_expanded = position_ids[..., None].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = position_ids_expanded * self.inv_freq.float()
            cos = freqs.cos() * self.attention_scaling
            sin = freqs.sin() * self.attention_scaling

        cos = self.recomposition_frequencies(cos)
        sin = self.recomposition_frequencies(sin)
        return cos, sin

    def recomposition_frequencies(self, freq):
        """
        Recompose the frequencies into the final spatial layout used per each grid.
        """
        freq_h, freq_w = freq[:, 0], freq[:, 1]
        freq_hw = torch.cat([freq_h, freq_w], dim=-1)
        return torch.cat([freq_hw, freq_hw], dim=-1)


class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = "silu"

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * ACT2FN[self.activation](gate.to(torch.float32))

        return hidden_states.to(input_dtype)


def causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: nn.Parameter,
    bias: nn.Parameter | None = None,
    activation: str | None = None,
):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = out[:, :, -seq_len:]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def causal_conv1d_fn(
    hidden_states: torch.Tensor,
    weight: nn.Parameter,
    bias: nn.Parameter | None = None,
    activation: str | None = None,
    **kwargs,
):
    _, hidden_size, seq_len = hidden_states.shape
    padding = weight.shape[-1] - 1

    out = F.conv1d(
        hidden_states.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=bias,
        padding=padding,
        groups=hidden_size,
    )[:, :, :seq_len]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


# NOTE: the FLA package computes `x / torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)` instead, so if we align
# with the GatedRMSNorm, maybe we can make that change as well.
def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
    padded_output_shape = (batch_size, num_v_heads, -1, v_head_dim)  # -1 is the padded sequence length
    decay = g  # rename for clarity: argument name must stay "g" to match flash_linear_attention's API

    # Make sure all tensors are fp32 and reshape them to [batch_size, num_*_heads, seqlen, ...]
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    # If enabled, normalize query and key vectors (in fp32 to match the FLA library)
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    # And always normalize queries by the head dimension
    scaling = query.shape[-1] ** -0.5
    query = query * scaling

    # Pad sequence length to be a multiple of chunk_size. Padding is described as (left_pad, right_pad) for each dim.
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))

    total_sequence_length = sequence_length + pad_size
    num_chunks = total_sequence_length // chunk_size

    # Apply beta to K and V, which is the "learning rate" of the recurrent state for a given token, ie. how much the new
    # state influence the old state. Beta is often normalized to (0, 1) where 0 = no update; 1 = overwrite old state.
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # Reshape all tensors to chunk the sequence dimension (adds a new dimension of size chunk_size)
    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)

    # Create a chunk-sized strictly upper triangular mask, ie. the mask of what a causal chunk may not attend to
    strictly_upper_mask = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)

    # Cumulative decay within each chunk (dim 3 is the position inside the chunk). Since decay is in log space,
    # cum_decay[..., t] is the log of a product of decays between the start of the chunk and position t
    cum_decay = decay.cumsum(dim=3)

    # First phase: compute intra-chunk quantities.
    # The pairwise decays: pairwise_decay[..., i, j] = exp(cum_decay_i - cum_decay_j) is the decay accumulated between
    # positions j and i of a chunk. Positive values are masked to -inf before exp to avoid overflow
    pairwise_decay = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
    pairwise_decay = pairwise_decay.masked_fill(strictly_upper_mask, float("-inf"))
    pairwise_decay = pairwise_decay.exp()  # with the exp, we exit log space, so we can apply this decay to the states

    # Compute auxiliary tensors: the Upper Triangular (ut) transform system and the intra-chunk attn (QK dot product)
    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

    # Gated delta attention uses a UT transform to condense several delta rule updates into a few matmuls. After the UT
    # system is solved, we can then compute the new_values (called "u" in the DeltaNet paper) and the decayed keys
    # reading the old state (k_cumdecay). In the update, the part of new_values that the old state already predicts is
    # subtracted out, so that only the correction is written to the recurrent state: this is the delta rule.

    # Not all export targets support the fast triangular solver, so we build the inverse by forward substitution then
    if not is_torchdynamo_exporting():
        new_values = torch.linalg.solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)
        k_cumdecay = torch.linalg.solve_triangular(ut_system, decayed_k_beta, upper=False, unitriangular=True)
    else:
        ut_system = -ut_system.tril(-1)  # ut_system is masked to only keep the strictly lower triangle
        for i in range(1, chunk_size):
            row = ut_system[..., i, :i].clone()
            sub = ut_system[..., :i, :i].clone()
            ut_system[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        ut_system = ut_system + torch.eye(chunk_size, dtype=ut_system.dtype, device=ut_system.device)
        new_values, k_cumdecay = ut_system @ v_beta, ut_system @ decayed_k_beta

    if initial_state is None:
        last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=new_values.dtype, device=new_values.device)
    else:
        last_recurrent_state = initial_state.to(new_values)
    core_attn_out = torch.zeros_like(new_values)

    # Apply decay once rather than in each chunk
    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]

    # Second phase: the sequential scan over chunks
    for i in range(num_chunks):
        # Compute attention output for the current chunk: add the read of the previous recurrent state
        # (inter_chunk_attn) with the within-chunk attention (intra_chunk_attn)
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ last_recurrent_state
        inter_chunk_attn = query[:, :, i] @ last_recurrent_state
        core_attn_out[:, :, i] = inter_chunk_attn + intra_chunk_attn[:, :, i] @ v_new
        # Update the recurrent state: new recurrent state (S_t+1) = decayed old state (S_t * (I-βkk^T)) + update (βvk^T)
        last_recurrent_state = last_recurrent_state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new

    # Discard the final state if not requested
    last_recurrent_state = None if not output_final_state else last_recurrent_state
    # Reshape the output to the orignal shape: flatten the chunk dimension, then drop padding
    core_attn_out = core_attn_out.reshape(padded_output_shape)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    # Convert back to the original shape [batch_size, sequence_length, num_v_heads, v_head_dim] and dtype
    core_attn_out = core_attn_out.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    decay = g  # rename for clarity: argument name must stay "g" to match flash_linear_attention's API

    # Make sure all tensors are fp32 and reshape them to [batch_size, num_*_heads, seqlen, ...]
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    # If enabled, normalize query and key vectors (done once in fp32 for better accuracy)
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    # And always normalize queries by the head dimension
    query = query / (query.shape[-1] ** 0.5)

    # Create the storage for the last recurrent state, which will be updated in place. If a previous state is provided,
    # it is the starting point, otherwise start with a zeroed buffer.
    if initial_state is None:
        recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
        last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=value.dtype, device=value.device)
    else:
        last_recurrent_state = initial_state.to(value)
    core_attn_out = torch.zeros_like(value)

    # Loop over each token and update the recurrent state
    for i in range(sequence_length):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        # Decay the recurrent state
        decay_t = decay[:, :, i].exp()[..., None, None]
        last_recurrent_state = last_recurrent_state * decay_t
        # Update the recurrent state with the current token
        beta_t = beta[:, :, i].unsqueeze(-1)
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # And use it to compute the attention output for the current token
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    # Discard the final state if not requested
    last_recurrent_state = None if not output_final_state else last_recurrent_state
    # Convert back to the original shape [batch_size, sequence_length, num_v_heads, v_head_dim] and dtype
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class Qwen3_5GatedDeltaNet(nn.Module):

    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        tp_size = dist.get_world_size()
        self.tp_size = tp_size
        self.tp_rank = dist.get_rank()
        self.hidden_size = config.hidden_size
        self.total_num_v_heads = config.linear_num_value_heads
        self.total_num_k_heads = config.linear_num_key_heads
        assert self.total_num_k_heads % tp_size == 0 and self.total_num_v_heads % tp_size == 0, (
            f"线性注意力的头数(k={self.total_num_k_heads}, v={self.total_num_v_heads})"
            f"必须都能被 tensor_parallel_size={tp_size} 整除")
        assert self.total_num_v_heads % self.total_num_k_heads == 0, (
            f"v 头数({self.total_num_v_heads})必须是 k 头数({self.total_num_k_heads})的整数倍(GQA 展开)")
        self.num_v_heads = self.total_num_v_heads // tp_size
        self.num_k_heads = self.total_num_k_heads // tp_size
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.total_key_dim = self.head_k_dim * self.total_num_k_heads
        self.total_value_dim = self.head_v_dim * self.total_num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.conv1d.weight.weight_loader = self.conv_weight_loader

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.dt_bias.weight_loader = self.v_head_weight_loader

        # Lower bound kept away from 0 so log(A) never becomes -inf
        A = torch.empty(self.num_v_heads).uniform_(0.01, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log.weight_loader = self.v_head_weight_loader

        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = RowParallelLinear(self.total_value_dim, self.hidden_size, bias=False)

        self.layer_type = config.layer_types[layer_idx]

        self.in_proj_qkv = MergedColumnParallelLinear(
            self.hidden_size,
            [self.total_key_dim, self.total_key_dim, self.total_value_dim],
            bias=False,
        )
        self.in_proj_qkv.weight.weight_loader = self.qkv_weight_loader
        self.in_proj_z = ColumnParallelLinear(self.hidden_size, self.total_value_dim, bias=False)
        self.in_proj_b = ColumnParallelLinear(self.hidden_size, self.total_num_v_heads, bias=False)
        self.in_proj_a = ColumnParallelLinear(self.hidden_size, self.total_num_v_heads, bias=False)

        self.state_shapes = (
            (self.conv_dim, self.conv_kernel_size - 1),
            (self.num_v_heads, self.head_k_dim, self.head_v_dim),
        )
        self.state_dtypes = (self.conv1d.weight.dtype, torch.float32)
        self.state_group = 0
        self.register_buffer("conv_pool", torch.empty(0), persistent=False)
        self.register_buffer("recurrent_pool", torch.empty(0), persistent=False)

    def _qkv_segment_sizes(self) -> tuple[int, int, int]:
        return self.total_key_dim, self.total_key_dim, self.total_value_dim

    def qkv_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        offset = 0
        for shard_id, size in enumerate(self._qkv_segment_sizes()):
            MergedColumnParallelLinear.weight_loader(
                self.in_proj_qkv, param, loaded_weight.narrow(0, offset, size), shard_id)
            offset += size

    def conv_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        pieces = []
        offset = 0
        for size in self._qkv_segment_sizes():
            local = size // self.tp_size
            pieces.append(loaded_weight.narrow(0, offset + self.tp_rank * local, local))
            offset += size
        param.data.copy_(torch.cat(pieces, dim=0))

    def v_head_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        local = self.num_v_heads
        param.data.copy_(loaded_weight.narrow(0, self.tp_rank * local, local))

    def _slot_ids(self) -> torch.Tensor:
        slot_ids = get_context().linear_slot_ids
        assert slot_ids is not None, (
            "线性注意力层需要序列的状态块 id(linear_slot_ids),但 context 里没有:"
            "序列进模型之前要先经过调度器的块分配")
        return slot_ids[:, self.state_group]

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        if get_context().is_prefill:
            return self.prefill(hidden_states)
        return self.decode(hidden_states)

    def _beta_g(self, hidden_states: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.in_proj_a(hidden_states).view(-1, seq_len, self.num_v_heads)
        beta = self.in_proj_b(hidden_states).view(-1, seq_len, self.num_v_heads).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        return beta, g

    def _split_qkv(self, mixed_qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = mixed_qkv.shape
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        return query, key, value

    def _finalize(
        self,
        core_attn_out: torch.Tensor,
        z: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """gated RMSNorm + out_proj: [B, T, num_v_heads, head_v_dim] -> [B, T, hidden]"""
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)

    def prefill(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        slot_ids = self._slot_ids().tolist()
        cu_seqlens_q = context.cu_seqlens_q.tolist()
        cu_seqlens_k = context.cu_seqlens_k.tolist()
        kernel_size = self.conv_kernel_size - 1
        outputs = []
        for i, slot in enumerate(slot_ids):
            q_start, q_end = cu_seqlens_q[i], cu_seqlens_q[i + 1]
            seq_len = q_end - q_start
            start = (cu_seqlens_k[i + 1] - cu_seqlens_k[i]) - seq_len
            assert start == 0, (
                f"第 {i} 条序列从第 {start} 个 token 开始,说明它复用了 KV cache(prefix cache 命中)"
                "或被切块调度,这两种情况下线性注意力的状态无法恢复。"
                "线性模型的 prefix cache 已在 BlockManager 里关掉;如果仍然触发,检查 max_num_batched_tokens "
                "是否小于 prompt 长度(切块 prefill 目前不支持线性层)。"
            )

            x = hidden_states[q_start:q_end].unsqueeze(0)                    # [1, T, hidden]
            mixed_qkv = self.in_proj_qkv(x).transpose(1, 2)                  # [1, conv_dim, T],卷积前的值
            conv_state = mixed_qkv[:, :, -kernel_size:]
            if conv_state.shape[-1] < kernel_size:
                conv_state = F.pad(conv_state, (kernel_size - conv_state.shape[-1], 0))
            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation=self.activation,
            ).transpose(1, 2)

            query, key_states, value = self._split_qkv(mixed_qkv)
            z = self.in_proj_z(x).reshape(1, seq_len, -1, self.head_v_dim)
            beta, g = self._beta_g(x, seq_len)

            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query,
                key_states,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            self.conv_pool[slot] = conv_state[0]
            self.recurrent_pool[slot] = last_recurrent_state[0]
            outputs.append(self._finalize(core_attn_out, z, 1, seq_len)[0])

        return torch.cat(outputs, dim=0)

    def decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        slot_ids = self._slot_ids()                                          # [B] 本组的状态块 id
        conv_states = self.conv_pool[slot_ids]                               # [B, conv_dim, k-1]
        mixed_qkv = self.in_proj_qkv(hidden_states).unsqueeze(1).transpose(1, 2)    # [B, conv_dim, 1]
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.activation,
        ).transpose(1, 2)
        self.conv_pool[slot_ids] = conv_states

        query, key_states, value = self._split_qkv(mixed_qkv)
        z = self.in_proj_z(hidden_states).reshape(-1, 1, self.num_v_heads, self.head_v_dim)
        beta, g = self._beta_g(hidden_states, 1)

        core_attn_out, last_recurrent_states = torch_recurrent_gated_delta_rule(
            query,
            key_states,
            value,
            g=g,
            beta=beta,
            initial_state=self.recurrent_pool[slot_ids],
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        self.recurrent_pool[slot_ids] = last_recurrent_states

        return self._finalize(core_attn_out, z, slot_ids.numel(), 1)[:, 0]


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Adapted from transformers.models.glm.modular_glm.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen3_5MRotaryEmbedding(nn.Module):

    def __init__(self, rotary_dim: int, base: float, mrope_section):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        freq_row = torch.zeros(inv_freq.numel(), dtype=torch.long)
        for dim, offset in enumerate((1, 2), start=1):      # dim=1 → H,dim=2 → W
            freq_row[offset : mrope_section[dim] * 3 : 3] = dim
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("freq_row", freq_row, persistent=False)

    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor):
        if positions.dim() == 1:    # 纯文本:三条流相同
            positions = positions.unsqueeze(0).expand(3, -1)
        
        freqs = positions.float()[:, :, None] * self.inv_freq[None, None, :]
        picked = freqs[0]
        for row in (1, 2):
            picked = torch.where(self.freq_row == row, freqs[row], picked)
        cos, sin = picked.cos().unsqueeze(1), picked.sin().unsqueeze(1)
        return apply_rotary_emb(query, cos, sin), apply_rotary_emb(key, cos, sin)


class Qwen3_5Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        
        tp_size = dist.get_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0, (
            f"注意力头数({self.total_num_heads})不能被 tensor_parallel_size={tp_size} 整除")
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_kv_heads % tp_size == 0, (
            f"KV 头数({self.total_num_kv_heads})不能被 tensor_parallel_size={tp_size} 整除:"
            f"这个模型的 TP 上限是 {self.total_num_kv_heads}")
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = config.head_dim or config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = config.attention_bias
        
        output_size = (2 * self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_dim
                
        self.qkv_proj = QKVParallelLinear(
            output_size,
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            q_gate=True,    # attn_output_gate:q_proj 里 q 和门控拼在一起(q 占两份宽度)
            bias=self.qkv_bias,
        )
        
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            config.attention_bias,
        )
        
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        
        rope_parameters = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
        rope_theta = rope_parameters.get("rope_theta", getattr(config, "rope_theta", 1000000))
        # partial_rotary_factor=0.25:只转 head_dim 的前 1/4,其余维度原样通过。
        self.rotary_dim = int(self.head_dim * rope_parameters.get("partial_rotary_factor", 1.0))
        self.rotary_emb = Qwen3_5MRotaryEmbedding(
            self.rotary_dim,
            rope_theta,
            rope_parameters.get("mrope_section") or [11, 11, 10],
        )
        
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        qkv = self.qkv_proj(hidden_states)
        query_gate, key_states, value_states = qkv.split(
            [2 * self.q_size, self.kv_size, self.kv_size], dim=-1)
        query_states, gate = query_gate.view(
            *input_shape, -1, self.head_dim * 2).chunk(2, dim=-1)
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape))
        key_states = self.k_norm(key_states.view(hidden_shape))
        value_states = value_states.view(hidden_shape)

        if self.rotary_dim == self.head_dim:
            query_states, key_states = self.rotary_emb(position_ids, query_states, key_states)
        else:
            query_rot, key_rot = self.rotary_emb(
                position_ids, query_states[..., : self.rotary_dim], key_states[..., : self.rotary_dim]
            )
            query_states = torch.cat([query_rot, query_states[..., self.rotary_dim :]], dim=-1)
            key_states = torch.cat([key_rot, key_states[..., self.rotary_dim :]], dim=-1)

        attn_output = self.attn(query_states, key_states, value_states)

        attn_output = attn_output.flatten(1, -1)
        attn_output = attn_output * torch.sigmoid(gate)

        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen3_5MLP(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, intermediate_size: int):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size

        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Qwen3_5 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.block_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Token Mixer
        if self.block_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        elif self.block_type == "full_attention":
            # Self Attention
            hidden_states = self.self_attn(hidden_states, position_ids=position_ids)

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3_5VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3_5VisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen3_5VisionPatchMerger(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig, use_postshuffle_norm=False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class Qwen3_5VisionAttention(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        max_seqlen: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface = eager_attention_forward

        if is_flash_attention_requested(self.config):
            # Flash Attention: Use cu_seqlens for variable length attention
            max_seqlen = get_max_seqlen(cu_seqlens, self.config, kwargs={"max_seqlen": max_seqlen})
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3_5VisionBlock(nn.Module):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3_5VisionAttention(config=config)
        self.mlp = Qwen3_5VisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        cu_seqlens (`torch.Tensor`):
            Cumulative sequence lengths used for packed variable-length attention in Flash Attention kernels.
        """
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3_5VisionModel(nn.Module):
    config: Qwen3_5VisionConfig

    def __init__(self, config, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3_5VisionPatchEmbed(
            config=config,
        )

        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        # How the (square) learned position grid is resampled to each image's grid.
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        self.interpolation_align_corners = True
        self.interpolation_mode = "bilinear"

        self.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(config)

        self.blocks = nn.ModuleList([Qwen3_5VisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3_5VisionPatchMerger(
            config=config,
            use_postshuffle_norm=False,
        )

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side=self.num_grid_per_side,
            mode=self.interpolation_mode,
            align_corners=self.interpolation_align_corners,
            spatial_merge_size=self.config.spatial_merge_size,
            kwargs=kwargs,
        )
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size, kwargs=kwargs)
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, self.config, kwargs=kwargs)

        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = (self.pos_embed(interp_indices) * interp_weights[:, :, None]).sum(1)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
        position_embeddings = self.rotary_pos_emb(hidden_states, position_ids)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        merged_hidden_states = self.merger(hidden_states)

        return merged_hidden_states


class Qwen3_5TextModel(nn.Module):
    config: Qwen3_5TextConfig

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.has_linear_attention = any(t == "linear_attention" for t in config.layer_types)
        self.linear_padding_slot: int | None = None

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_ids=position_ids,
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states


class Qwen3_5Model(nn.Module):
    base_model_prefix = "model"
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.visual = Qwen3_5VisionModel(config.vision_config)
        self.language_model = Qwen3_5TextModel(config.text_config)
        self.rope_deltas = None 

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: list[int, int, int] | torch.Tensor,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device: str | torch.device | None = None,
    ):
        llm_grid_t, llm_grid_h, llm_grid_w = (
            grid_thw[0].item() // temp_merge_size,
            grid_thw[1].item() // spatial_merge_size,
            grid_thw[2].item() // spatial_merge_size,
        )

        position_temporal = torch.arange(llm_grid_t, device=device) * time_interval
        position_height = torch.arange(llm_grid_h, device=device) + start_position
        position_width = torch.arange(llm_grid_w, device=device) + start_position

        T_grid, H_grid, W_grid = torch.meshgrid(position_temporal, position_height, position_width, indexing="ij")
        vision_position_ids = torch.stack([T_grid, H_grid, W_grid], dim=0).reshape(3, -1)
        vision_position_ids[0] += start_position
        return vision_position_ids

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1
        spatial_merge_size = self.config.vision_config.spatial_merge_size

        mrope_position_deltas = []
        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }

        for batch_idx, current_input_ids in enumerate(input_ids):
            input_token_type = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
                input_token_type = input_token_type[attention_mask[batch_idx].bool()]

            input_type_group = []
            for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
                group = list(group)
                start_index = group[0][0]
                end_index = group[-1][0] + 1
                input_type_group.append((key, start_index, end_index))

            current_pos = 0
            llm_pos_ids_list = []
            for modality_type, start_idx, end_idx in input_type_group:
                # text == 0
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                    )
                    current_pos += text_len
                # image == 1, video == 2
                else:
                    grid_thw = next(grid_iters[modality_type])
                    vision_position_ids = self.get_vision_position_ids(
                        current_pos, grid_thw, 1, spatial_merge_size, device=input_ids.device
                    )
                    llm_pos_ids_list.append(vision_position_ids)
                    current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas


    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs,
    ):
        return self.get_image_features(pixel_values_videos, video_grid_thw, **kwargs)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs,
    ):
        pixel_values = pixel_values.type(self.visual.patch_embed.proj.weight.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw, **kwargs)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)

        return image_embeds

    def _vision_features_parallel(self, get_features, pixel_values, grid_thw, **kwargs) -> torch.Tensor:
        world = dist.get_world_size()
        rank = dist.get_rank()
        rows = grid_thw.prod(-1).tolist()
        shards = plan_shards(rows, world)
        tokens = (grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        offsets, total = [], 0
        for shard in shards:
            n = sum(tokens[shard.item_start:shard.item_end])
            offsets.append((total, total + n))
            total += n

        full = torch.empty(total, self.config.vision_config.out_hidden_size,
                           dtype=self.visual.patch_embed.proj.weight.dtype,
                           device=pixel_values.device)
        mine = shards[rank]
        if mine.row_end > mine.row_start:
            embeds = get_features(pixel_values, grid_thw[mine.item_start:mine.item_end], **kwargs)
            full[offsets[rank][0]:offsets[rank][1]] = torch.cat(embeds, dim=0)

        if world > 1:
            for src, (start, end) in enumerate(offsets):
                if end > start:
                    dist.broadcast(full[start:end], src=src)
        return full

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
        video_features: torch.FloatTensor | None = None,
    ):
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.full((), self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.full((), self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).to(inputs_embeds.device)
        if image_features is not None:
            torch_compilable_check(
                n_image_tokens * inputs_embeds.shape[-1] == image_features.numel(),
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {image_features.shape[0]}",
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).to(inputs_embeds.device)
        if video_features is not None:
            torch_compilable_check(
                n_video_tokens * inputs_embeds.shape[-1] == video_features.numel(),
                f"Video features and video tokens do not match, tokens: {n_video_tokens}, features: {video_features.shape[0]}",
            )
        return special_image_mask, special_video_mask

    def compute_3d_position_ids(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: torch.Tensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
    ) -> torch.Tensor | None:
        past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        has_multimodal = image_grid_thw is not None or video_grid_thw is not None
        if has_multimodal and mm_token_type_ids is None and input_ids is not None:
            raise ValueError(
                "Multimodal data was passed (via `image_grid_thw` or `video_grid_thw`) but `mm_token_type_ids` is "
                "missing. Please pass `mm_token_type_ids` to the model so that multimodal RoPE (M-RoPE) can be "
                "computed correctly. `mm_token_type_ids` is returned by the processor alongside `input_ids`."
            )
        can_compute_mrope = input_ids is not None and mm_token_type_ids is not None and has_multimodal

        if can_compute_mrope and (self.rope_deltas is None or past_key_values_length == 0):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
            )
            self.rope_deltas = rope_deltas
        elif self.rope_deltas is not None and (past_key_values_length > 0 or input_ids is None):
            batch_size, seq_length, _ = inputs_embeds.shape
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.masked_fill(attention_mask == 0, 0)
                position_ids = position_ids.view(1, batch_size, -1).repeat(3, 1, 1).to(inputs_embeds.device)
            else:
                position_ids = torch.arange(past_key_values_length, past_key_values_length + seq_length)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1).to(inputs_embeds.device)
            delta = self.rope_deltas.repeat_interleave(batch_size // self.rope_deltas.shape[0], dim=0)
            position_ids = position_ids + delta.to(device=inputs_embeds.device)
        else:
            position_ids = None
        return position_ids
    
    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs,
    ):
        inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self._vision_features_parallel(
                self.get_image_features, pixel_values, image_grid_thw, **kwargs)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self._vision_features_parallel(
                self.get_video_features, pixel_values_videos, video_grid_thw, **kwargs)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        position_ids = torch.arange(inputs_embeds.shape[0], device=inputs_embeds.device)

        return self.language_model(
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )


class Qwen3_5ForCausalLM(nn.Module):
    ignore_weight_prefixes = ("mtp.",)
    
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config):
        super().__init__()
        self.model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(config.text_config.vocab_size, config.text_config.hidden_size)
        
        if config.text_config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.language_model.embed_tokens.weight.data

    @property
    def mm_pixel_dtype(self):
        return self.model.visual.patch_embed.proj.weight.dtype

    @property
    def linear_padding_slot(self) -> int | None:
        """线性注意力状态池的占位块 id(模型没有线性注意力层时为 None),见 Qwen3_5TextModel。"""
        return self.model.language_model.linear_padding_slot

    @linear_padding_slot.setter
    def linear_padding_slot(self, value: int | None):
        self.model.language_model.linear_padding_slot = value

    def mm_position_ids(self, seq) -> torch.LongTensor:
        mm = seq.mm
        input_ids = torch.tensor([seq.prompt_token_ids], dtype=torch.long, device="cpu")
        mm_token_type_ids = torch.tensor([mm.mm_token_type_ids], dtype=torch.long, device="cpu")
        image_grid_thw = None if mm.image_grid_thw is None else mm.image_grid_thw.cpu()
        video_grid_thw = None if mm.video_grid_thw is None else mm.video_grid_thw.cpu().clone()
        positions, deltas = self.model.get_rope_index(
            input_ids,
            mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        mm.rope_delta = int(deltas.flatten()[0])
        return positions[:, 0]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        **kwargs,
    ):
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )

        return hidden_states
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)