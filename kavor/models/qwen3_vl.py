import itertools

import torch
import torch.distributed as dist
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func
from torch import nn
from transformers import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig
from transformers.activations import ACT2FN
from transformers.vision_utils import (
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

from kavor.layers.activation import SiluAndMul
from kavor.layers.attention.flash_attention_kvcache import Attention
from kavor.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from kavor.layers.layernorm import RMSNorm
from kavor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from kavor.layers.rotary_embedding import apply_rotary_emb
from kavor.utils.mm_plan import plan_shards, visible_row_index


class Qwen3VLVisionMLP(nn.Module):

    def __init__(self, config: Qwen3VLVisionConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3VLVisionPatchEmbed(nn.Module):

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        return self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    """视觉塔的 2D axial RoPE:H、W 各用一份 `head_dim//4` 的频率,最后拼成整个 head_dim。"""

    def __init__(self, config: Qwen3VLVisionConfig, device=None):
        super().__init__()
        self.config = config

        self.rope_type = config.rope_parameters["rope_type"]
        if self.rope_type != "axial":
            raise ValueError(
                f"{self.__class__.__name__} supports only axial rope, but requested {self.rope_type}")
        inv_freq, self.attention_scaling = self.compute_axial_rope_parameters(config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def compute_axial_rope_parameters(
        config: Qwen3VLVisionConfig, device=None, **kwargs
    ) -> tuple[torch.Tensor, float]:
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_heads
        spatial_dim = dim // 2

        attention_factor = 1.0  # 这类 RoPE 不用缩放
        inv_freq = 1.0 / (base ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim))
        return inv_freq.to(device), attention_factor

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        # position_ids: (2, N),第 0 行是 h 坐标,第 1 行是 w 坐标
        position_ids_expanded = position_ids[..., None].float()
        freqs = position_ids_expanded * self.inv_freq.float()
        cos = freqs.cos() * self.attention_scaling
        sin = freqs.sin() * self.attention_scaling

        cos = self.recomposition_frequencies(cos)
        sin = self.recomposition_frequencies(sin)
        return cos, sin

    @staticmethod
    def recomposition_frequencies(freq: torch.Tensor) -> torch.Tensor:
        freq_h, freq_w = freq[:, 0], freq[:, 1]
        freq_hw = torch.cat([freq_h, freq_w], dim=-1)
        return torch.cat([freq_hw, freq_hw], dim=-1)


class Qwen3VLVisionPatchMerger(nn.Module):

    def __init__(self, config: Qwen3VLVisionConfig, use_postshuffle_norm: bool = False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """把输入的后半维旋转到前面(rotate-half 形式的 RoPE)。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class Qwen3VLVisionAttention(nn.Module):

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=self.scaling,
            causal=False,
        )
        return self.proj(attn_output.reshape(seq_length, -1).contiguous())


class Qwen3VLVisionBlock(nn.Module):

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config)
        self.mlp = Qwen3VLVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VLVisionModel(nn.Module):

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = config.spatial_merge_size ** 2
        self.patch_size = config.patch_size

        self.patch_embed = Qwen3VLVisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        # 把方形的位置网格双线性重采样到每张图自己的 (h, w) 网格
        self.num_grid_per_side = int(config.num_position_embeddings ** 0.5)
        self.interpolation_align_corners = True
        self.interpolation_mode = "bilinear"

        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(config)

        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=False)

        self.deepstack_visual_indexes = tuple(config.deepstack_visual_indexes)
        self.deepstack_merger_list = nn.ModuleList(
            [Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=True)
             for _ in range(len(self.deepstack_visual_indexes))]
        )

    @staticmethod
    def attention_seqlens(grid_thw: torch.Tensor) -> tuple[torch.Tensor, int]:
        """packed attention 的段边界:每个时域 patch 单独一段,长度 h*w。"""
        seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
        cu_seqlens = F.pad(seqlens.cumsum(0, dtype=torch.int32), (1, 0), value=0)
        return cu_seqlens, int(seqlens.max())

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """返回 (merger 之后的图像特征, 各 deepstack 层的特征)。

        两份特征都按本批图的顺序拼接,行数也已经过 `spatial_merge_size ** 2` 合并,
        和文本里图像占位符的个数对应(deepstack merger 同样做了 spatial merge)。
        """
        interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side=self.num_grid_per_side,
            mode=self.interpolation_mode,
            align_corners=self.interpolation_align_corners,
            spatial_merge_size=self.spatial_merge_size,
        )
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens, max_seqlen = self.attention_seqlens(grid_thw)

        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = (self.pos_embed(interp_indices) * interp_weights[:, :, None]).sum(1)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
        position_embeddings = self.rotary_pos_emb(hidden_states, position_ids)

        deepstack_features: list[torch.Tensor] = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.deepstack_visual_indexes:
                merger = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)]
                deepstack_features.append(merger(hidden_states))

        return self.merger(hidden_states), deepstack_features


# ---------------------------------------------------------------------------
# 文本塔
# ---------------------------------------------------------------------------


class Qwen3VLMRotaryEmbedding(nn.Module):

    def __init__(self, head_dim: int, base: float, mrope_section: list[int]):
        super().__init__()
        assert sum(mrope_section) == head_dim // 2, (
            f"mrope_section {mrope_section} 之和必须等于 head_dim/2 = {head_dim // 2}")

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        freq_row = torch.zeros(inv_freq.numel(), dtype=torch.long)
        for dim, offset in enumerate((1, 2), start=1):      # dim=1 -> H, dim=2 -> W
            freq_row[offset: mrope_section[dim] * 3: 3] = dim

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("freq_row", freq_row, persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.dim() == 1:      # 纯文本 / 解码步:三条流相同
            positions = positions.unsqueeze(0).expand(3, -1)
        assert positions.shape[0] == 3, (
            f"位置必须是 (T,H,W) 三行(packed 批次)或 1 维(解码步),收到 {tuple(positions.shape)}")

        freqs = positions.float()[:, :, None] * self.inv_freq[None, None, :]
        picked = freqs[0]
        for row in (1, 2):
            picked = torch.where(self.freq_row == row, freqs[row], picked)
        # (N, 1, head_dim/2),广播到 (N, num_heads, head_dim/2)
        return picked.cos().unsqueeze(1), picked.sin().unsqueeze(1)


class Qwen3VLTextAttention(nn.Module):

    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
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

        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        output_size = (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_dim
        self.qkv_proj = QKVParallelLinear(
            output_size,
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1)
        query_states = self.q_norm(query_states.view(-1, self.num_heads, self.head_dim))
        key_states = self.k_norm(key_states.view(-1, self.num_kv_heads, self.head_dim))
        value_states = value_states.view(-1, self.num_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states = apply_rotary_emb(query_states, cos, sin)
        key_states = apply_rotary_emb(key_states, cos, sin)

        attn_output = self.attn(query_states, key_states, value_states)
        return self.o_proj(attn_output.flatten(1, -1))


class Qwen3VLTextMLP(nn.Module):

    def __init__(self, config: Qwen3VLTextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        assert config.hidden_act == "silu"
        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size, [self.intermediate_size] * 2, bias=False)
        self.down_proj = RowParallelLinear(
            self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Qwen3VLTextDecoderLayer(nn.Module):

    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3VLTextAttention(config, layer_idx)
        self.mlp = Qwen3VLTextMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


class Qwen3VLTextModel(nn.Module):

    def __init__(self, config: Qwen3VLTextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3VLTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        rope_parameters = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
        rope_theta = rope_parameters.get("rope_theta", getattr(config, "rope_theta", 5000000))
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.rotary_emb = Qwen3VLMRotaryEmbedding(
            head_dim, rope_theta, rope_parameters.get("mrope_section") or [24, 20, 20])

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        cos, sin = self.rotary_emb(position_ids)
        cos, sin = cos.to(inputs_embeds.dtype), sin.to(inputs_embeds.dtype)
        position_embeddings = (cos, sin)

        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(hidden_states, position_embeddings)
            # 前几层把视觉特征加回残差流(DeepStack)
            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                hidden_states = self._deepstack_process(
                    hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx])

        return self.norm(hidden_states)

    @staticmethod
    def _deepstack_process(
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        hidden_states[visual_pos_masks, :] += visual_embeds
        return hidden_states


class Qwen3VLModel(nn.Module):
    base_model_prefix = "model"
    accepts_loss_kwargs = False

    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        self.visual = Qwen3VLVisionModel(config.vision_config)
        self.language_model = Qwen3VLTextModel(config.text_config)
        self.spatial_merge_size = config.vision_config.spatial_merge_size

    @staticmethod
    def get_vision_position_ids(
        start_position: int,
        grid_thw: torch.Tensor,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """单张图/一段视频的 3D 位置(temporal, height, width),展平后偏移 `start_position`。"""
        llm_grid_t = grid_thw[0].item() // temp_merge_size
        llm_grid_h = grid_thw[1].item() // spatial_merge_size
        llm_grid_w = grid_thw[2].item() // spatial_merge_size

        position_temporal = torch.arange(llm_grid_t, device=device) * time_interval
        position_height = torch.arange(llm_grid_h, device=device) + start_position
        position_width = torch.arange(llm_grid_w, device=device) + start_position

        t_grid, h_grid, w_grid = torch.meshgrid(
            position_temporal, position_height, position_width, indexing="ij")
        vision_position_ids = torch.stack([t_grid, h_grid, w_grid], dim=0).reshape(3, -1)
        vision_position_ids[0] += start_position      # 必须在乘 time_interval 之后
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
                input_type_group.append((key, group[0][0], group[-1][0] + 1))

            current_pos = 0
            llm_pos_ids_list = []
            for modality_type, start_idx, end_idx in input_type_group:
                if modality_type == 0:      # text
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1)
                        + current_pos)
                    current_pos += text_len
                else:                        # image == 1, video == 2
                    grid_thw = next(grid_iters[modality_type])
                    llm_pos_ids_list.append(self.get_vision_position_ids(
                        current_pos, grid_thw, 1, spatial_merge_size, device=input_ids.device))
                    current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(
                    position_ids.device)
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))

        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    def get_image_features(
        self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor
    ) -> list[torch.Tensor]:
        """非分片路径:一次算完整批图,返回按图切开的列表(和 Qwen3_5 的接口一致)。"""
        pixel_values = pixel_values.to(self.visual.patch_embed.proj.weight.dtype)
        merged, _ = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // (self.spatial_merge_size ** 2)).tolist()
        return list(torch.split(merged, split_sizes))

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: torch.LongTensor
    ) -> list[torch.Tensor]:
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    def get_mm_embeddings(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, modality: str = "image"
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """引擎侧缓存接口:只跑视觉塔,返回 (merged 特征, deepstack 特征列表)。

        图像和视频走同一条塔,`modality` 只为和 Qwen3_5 保持同签名;
        TP 分片与 broadcast 留在模型内部,调用方不需要知道 plan_shards。
        """
        del modality
        return self._vision_features_parallel(pixel_values, grid_thw)

    def _vision_features_parallel(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        vision_config = self.config.vision_config
        merge_unit = self.spatial_merge_size ** 2

        rows = grid_thw.prod(-1).tolist()
        shards = plan_shards(rows, world_size)
        merged_tokens = [r // merge_unit for r in rows]

        cum_merged = [0]
        for t in merged_tokens:
            cum_merged.append(cum_merged[-1] + t)

        out_hidden = vision_config.out_hidden_size
        dtype = self.visual.patch_embed.proj.weight.dtype
        device = pixel_values.device
        merged_full = torch.empty(cum_merged[-1], out_hidden, dtype=dtype, device=device)
        # deepstack merger 也做 spatial merge,所以行数和 merger 输出一致
        deepstack_full = [
            torch.empty(cum_merged[-1], out_hidden, dtype=dtype, device=device)
            for _ in self.visual.deepstack_visual_indexes
        ]

        mine = shards[rank]
        if mine.item_end > mine.item_start:
            merged, deepstack = self.visual(
                pixel_values.to(dtype), grid_thw=grid_thw[mine.item_start:mine.item_end])
            merged_full[cum_merged[mine.item_start]:cum_merged[mine.item_end]] = merged
            for buf, part in zip(deepstack_full, deepstack):
                buf[cum_merged[mine.item_start]:cum_merged[mine.item_end]] = part

        if world_size > 1:
            for src, shard in enumerate(shards):
                m_start, m_end = cum_merged[shard.item_start], cum_merged[shard.item_end]
                if m_end > m_start:
                    dist.broadcast(merged_full[m_start:m_end], src=src)
                    for buf in deepstack_full:
                        dist.broadcast(buf[m_start:m_end], src=src)

        return merged_full, deepstack_full

    def _select_visible_rows(
        self,
        embeds: torch.Tensor,
        deepstack_embeds: list[torch.Tensor],
        grid_thw: torch.Tensor,
        visible_ranges: list[tuple[int, int]] | None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        rows = (grid_thw.prod(-1) // (self.spatial_merge_size ** 2)).tolist()
        keep = visible_row_index(rows, visible_ranges)
        if keep is None:
            return embeds, deepstack_embeds
        index = torch.tensor(keep, dtype=torch.long, device=embeds.device)
        return embeds.index_select(0, index), [e.index_select(0, index) for e in deepstack_embeds]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        image_visible_ranges: list[tuple[int, int]] | None = None,
        video_visible_ranges: list[tuple[int, int]] | None = None,
        image_embeds: torch.Tensor | None = None,
        deepstack_image_embeds: list[torch.Tensor] | None = None,
        video_embeds: torch.Tensor | None = None,
        deepstack_video_embeds: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)
        has_media = (pixel_values is not None or pixel_values_videos is not None
                     or image_embeds is not None or video_embeds is not None)
        if has_media and input_ids is None:
            raise ValueError("多模态输入需要 input_ids:占位符位置是从 token id 里找的")

        if position_ids is None:
            if has_media:
                raise ValueError(
                    "多模态输入的 3D position_ids 需要由外部传入:"
                    "引擎里由 get_rope_index/mm_position_ids 预计算")
            position_ids = torch.arange(inputs_embeds.shape[0], device=inputs_embeds.device)

        image_mask = None
        video_mask = None

        if image_embeds is not None:
            # 引擎侧 mm cache 命中/未命中拼好的特征,视觉塔已经跑过了
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, deepstack_image_embeds = self._place_vision_tokens(
                input_ids, inputs_embeds, image_embeds, deepstack_image_embeds, "image")
            if image_mask is not None:
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        elif pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("传了 pixel_values 就要同时传 image_grid_thw")
            # 本步一个图片占位符都没有(整张图都在 prefix cache 里):视觉塔不用跑
            if (input_ids == self.config.image_token_id).any():
                image_embeds, deepstack_image_embeds = self._vision_features_parallel(
                    pixel_values, image_grid_thw)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                image_embeds, deepstack_image_embeds = self._select_visible_rows(
                    image_embeds, deepstack_image_embeds, image_grid_thw, image_visible_ranges)
                image_mask, deepstack_image_embeds = self._place_vision_tokens(
                    input_ids, inputs_embeds, image_embeds, deepstack_image_embeds, "image")
                if image_mask is not None:
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if video_embeds is not None:
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            video_mask, deepstack_video_embeds = self._place_vision_tokens(
                input_ids, inputs_embeds, video_embeds, deepstack_video_embeds, "video")
            if video_mask is not None:
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        elif pixel_values_videos is not None:
            if video_grid_thw is None:
                raise ValueError("传了 pixel_values_videos 就要同时传 video_grid_thw")
            if (input_ids == self.config.video_token_id).any():
                video_embeds, deepstack_video_embeds = self._vision_features_parallel(
                    pixel_values_videos, video_grid_thw)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                video_embeds, deepstack_video_embeds = self._select_visible_rows(
                    video_embeds, deepstack_video_embeds, video_grid_thw, video_visible_ranges)
                video_mask, deepstack_video_embeds = self._place_vision_tokens(
                    input_ids, inputs_embeds, video_embeds, deepstack_video_embeds, "video")
                if video_mask is not None:
                    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # 图 + 视频同时存在:按视觉 token 的先后顺序把两份 deepstack 特征合并
            image_pos = image_mask[..., 0]
            video_pos = video_mask[..., 0]
            visual_pos_masks = image_pos | video_pos
            image_mask_joint = image_pos[visual_pos_masks]
            video_mask_joint = video_pos[visual_pos_masks]
            deepstack_visual_embeds = []
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(
                    int(visual_pos_masks.sum()), img_embed.shape[-1])
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            visual_pos_masks = image_mask[..., 0]
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            visual_pos_masks = video_mask[..., 0]
            deepstack_visual_embeds = deepstack_video_embeds

        return self.language_model(
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

    def _place_vision_tokens(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        vision_embeds: torch.Tensor,
        deepstack_embeds: list[torch.Tensor],
        name: str,
    ) -> tuple[torch.Tensor | None, list[torch.Tensor] | None]:
        token_id = self.config.image_token_id if name == "image" else self.config.video_token_id
        mask = input_ids == token_id
        n_tokens = int(mask.sum())
        if n_tokens == 0:
            return None, None
        if n_tokens * inputs_embeds.shape[-1] != vision_embeds.numel():
            raise ValueError(
                f"{name} 特征和占位符对不上: 本步有 {n_tokens} 个占位符,"
                f"但特征有 {vision_embeds.numel() // inputs_embeds.shape[-1]} 个 token。"
                f"检查 processor 的输入;也可能是 {name}_visible_ranges 算错了")
        return mask.unsqueeze(-1).to(inputs_embeds.device), deepstack_embeds


class Qwen3VLForCausalLM(nn.Module):

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.model = Qwen3VLModel(config)
        self.lm_head = ParallelLMHead(config.text_config.vocab_size, config.text_config.hidden_size)
        if config.text_config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.language_model.embed_tokens.weight.data

    @property
    def mm_pixel_dtype(self) -> torch.dtype:
        """引擎按这个 dtype 把像素张量发给各 rank,省一次无用的大张量转换。"""
        return self.model.visual.patch_embed.proj.weight.dtype

    def get_mm_embeddings(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor,
                          modality: str = "image") -> tuple[torch.Tensor, list[torch.Tensor]]:
        """给 ModelRunner 的 mm cache 用:只算视觉塔,不做占位符填充。"""
        return self.model.get_mm_embeddings(pixel_values, grid_thw, modality)

    def mm_position_ids(self, seq) -> torch.LongTensor:
        """为一条多模态序列预计算 3D M-RoPE 位置,并把 rope_delta 回填到输入上。"""
        inputs = seq.mm_inputs
        assert inputs.mm_token_type_ids is not None, (
            "多模态序列缺少 mm_token_type_ids:processor 要和 input_ids 一起返回它,"
            "M-RoPE 才能算对")
        input_ids = torch.tensor([seq.prompt_token_ids], dtype=torch.long)
        mm_token_type_ids = torch.tensor([inputs.mm_token_type_ids], dtype=torch.long)
        image_grid_thw = (None if inputs.image is None or inputs.image.grid_thw is None
                          else inputs.image.grid_thw.cpu())
        video_grid_thw = (None if inputs.video is None or inputs.video.grid_thw is None
                          else inputs.video.grid_thw.cpu().clone())
        positions, deltas = self.model.get_rope_index(
            input_ids,
            mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        inputs.rope_delta = int(deltas.flatten()[0])
        return positions[:, 0]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
