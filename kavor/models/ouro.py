import torch
from torch import nn
import torch.distributed as dist

from kavor.layers.activation import SiluAndMul
from kavor.layers.attention import Attention
from kavor.layers.layernorm import RMSNorm
from kavor.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from kavor.layers.rotary_embedding import get_rope
from kavor.layers.embed_head import VocabParallelEmbedding, ParallelLMHead

from .configuration_auto import KavorConfig, register_config


@register_config("ouro")
class OuroConfig(KavorConfig):

    model_type = "ouro"

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        layer_types=None,
        attention_dropout=0.0,
        total_ut_steps=4,
        early_exit_threshold=1.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        self.total_ut_steps = total_ut_steps
        self.early_exit_threshold = early_exit_threshold
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        assert len(self.layer_types) == self.num_hidden_layers, (
            f"layer_types 长度 {len(self.layer_types)} 与 num_hidden_layers "
            f"{self.num_hidden_layers} 不一致")

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        if getattr(self, "early_exit_step", None) is not None or self.early_exit_threshold != 1.0:
            raise NotImplementedError(
                f"Ouro 的自适应退出还没实现:early_exit_step={getattr(self, 'early_exit_step', None)}, "
                f"early_exit_threshold={self.early_exit_threshold}。"
                f"当前只支持 early_exit_step 不设置且 early_exit_threshold=1.0")


class OuroMLP(nn.Module):
    def __init__(self, config: OuroConfig):
        
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

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
        assert config.hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class OuroAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: OuroConfig, layer_idx: int):
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_kv_heads % tp_size == 0
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
        )
        
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        
        rope_parameters = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
        rope_theta = rope_parameters.get("rope_theta", getattr(config, "rope_theta", 1000000))
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta
        )
        
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            num_kv_slots=config.total_ut_steps,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ut_step: int = 0,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v, ut_step)
        output = self.o_proj(o.flatten(1, -1))

        return output


class OuroDecoderLayer(nn.Module):
    def __init__(self, config: OuroConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = OuroAttention(config=config, layer_idx=layer_idx)

        self.mlp = OuroMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_2 = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm_2 = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        ut_step: int = 0,
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn(
            position_ids,
            hidden_states,
            ut_step,
        )
        hidden_states = self.input_layernorm_2(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_attention_layernorm_2(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class OuroModel(nn.Module):
    def __init__(
        self, 
        config: OuroConfig
    ):
        super().__init__()
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                OuroDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.total_ut_steps = getattr(config, "total_ut_steps", 4)
        self.early_exit_gate = RowParallelLinear(config.hidden_size, 1, bias=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:

        hidden_states = self.embed_tokens(input_ids)

        # 同一组层跑 total_ut_steps 轮,每轮都被 norm 后喂给下一轮;
        # early_exit_threshold=1.0 时自适应退出退化为"永远用最后一轮",所以这里直接返回最后一轮
        for ut_step in range(self.total_ut_steps):
            for decoder_layer in self.layers:
                hidden_states = decoder_layer(
                    hidden_states,
                    position_ids,
                    ut_step,
                )

            hidden_states = self.norm(hidden_states)

        return hidden_states


class OuroForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self, 
        config: OuroConfig
    ):
        super().__init__()
        self.model = OuroModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        return self.model(input_ids, position_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
