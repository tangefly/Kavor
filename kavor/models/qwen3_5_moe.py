"""Qwen3.5 / Qwen3.6 MoE(qwen3_5_moe)在 Kavor 上的实现。

架构上它和 ``kavor/models/qwen3_5.py`` 只差 FFN:线性注意力(GDN)+ full
attention 混合层、MRoPE、视觉塔、norm 全部复用那边的实现,这里只实现 MoE
块本身:

    y = Σ_{j∈top-k} w_j · Expert_j(x) + sigmoid(shared_gate(x)) · SharedExpert(x)

checkpoint 的专家是**按专家分散的 2D fp8 权重**
(``layers.N.mlp.experts.<e>.{gate,up,down}_proj``),而前向要走分组 GEMM
(``kavor/kernels/moe.py``),所以加载期把它们拼成按专家堆叠的 3D 参数
(TP 切中间维),由 ``resolve_stacked_weight`` 接管这些 key。
"""

import re

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from kavor.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from kavor.layers.linear import ReplicatedLinear
from kavor.layers.quant.base import UnquantizedMoeMethod
from kavor.models.qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5ForCausalLM,
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP,
    Qwen3_5Model,
    Qwen3_5RMSNorm,
    Qwen3_5TextModel,
    Qwen3_5VisionModel,
)


class Qwen3_5MoeTopKRouter(nn.Module):
    """router:bf16 打分 + fp32 softmax/top-k + 重归一化。

    Qwen3.5-MoE 的 config 里**没有** ``norm_topk_prob`` 字段(HF 用
    ``AttributeError()`` 显式删掉了),top-k 概率总是重归一化,别按
    Qwen3-MoE 的习惯去读那个属性。
    """

    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states: torch.Tensor):
        """(T, H) -> routing_weights (T, k) bf16, selected_experts (T, k) int32"""
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = torch.softmax(router_logits.float(), dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        return router_top_value.to(hidden_states.dtype), router_indices.to(torch.int32)


class Qwen3_5MoeExperts(nn.Module):
    """专家集合:按专家堆叠的 3D 权重 + per-expert checkpoint 加载。

    参数(TP 的 local_inter = moe_intermediate_size / tp_size,切的是中间维):
    - ``gate_up_proj``       (E, 2*local_inter, H):前一半 gate、后一半 up
    - ``down_proj``          (E, H, local_inter)
    - fp8 时另有 ``gate_up_scale_inv`` / ``down_scale_inv``(128x128 block scale)

    前向走 ``kavor/kernels/moe.py`` 的分组 GEMM,TP>1 时对专家部分做一次
    ``all_reduce``(专家各自线性,先加后 reduce 等价)。
    """

    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()
        assert self.intermediate_size % self.tp_size == 0, (
            f"moe_intermediate_size({self.intermediate_size})不能被 "
            f"tensor_parallel_size={self.tp_size} 整除")
        self.local_intermediate = self.intermediate_size // self.tp_size
        self.quant_method = (
            quant_config.get_moe_method(prefix)
            if quant_config is not None else UnquantizedMoeMethod())
        self.quant_method.create_weights(
            self, self.num_experts, self.local_intermediate,
            self.hidden_size, self.weight_loader)

    # ---------------- 权重加载 ----------------

    def _intermediate_slice(self, shard_id: int, unit: int) -> tuple[int, int]:
        """gate/up 各自的本地行区间(单位:元素或 128 宽的 block)。"""
        local = self.local_intermediate // unit
        offset = (0 if shard_id == 0 else self.local_intermediate) // unit
        return offset, local

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """融合 3D checkpoint(``experts.gate_up_proj``)的整表加载。"""
        if param is self.gate_up_proj or param is self.gate_up_scale_inv:
            unit = 1 if param is self.gate_up_proj else 128
            pieces = []
            for shard_id in (0, 1):
                offset, local = self._intermediate_slice(shard_id, unit)
                start = self.tp_rank * local + offset      # 两者都以 unit 为单位
                pieces.append(loaded_weight.narrow(1, start, local))
            param.data.copy_(torch.cat(pieces, dim=1))
        elif param is self.down_proj or param is self.down_scale_inv:
            unit = 1 if param is self.down_proj else 128
            local = self.local_intermediate // unit
            param.data.copy_(
                loaded_weight.narrow(2, self.tp_rank * local, local))
        else:
            raise RuntimeError(f"未知的专家参数:{param.shape}")

    def load_expert_tensor(
        self, proj: str, kind: str, expert: int, loaded_weight: torch.Tensor,
    ) -> str:
        """checkpoint 的 per-expert 2D 权重写进 3D 参数的对应专家行。

        返回被写入的参数名后缀(供 AutoModelForCausalLM 记录已加载参数)。
        """
        scale = kind == "weight_scale_inv"
        unit = 128 if scale else 1
        local = self.local_intermediate // unit
        start = self.tp_rank * local
        expert = int(expert)

        is_down = proj == "down_proj"
        if is_down:
            attr = "down_scale_inv" if scale else "down_proj"
        else:
            attr = "gate_up_scale_inv" if scale else "gate_up_proj"
        param = getattr(self, attr)

        if is_down:
            assert param is not None, f"{proj}.{kind} 出现但专家权重未量化"
            dst, src = param.data[expert], loaded_weight.narrow(1, start, local)
        else:
            assert param is not None, f"{proj}.{kind} 出现但专家权重未量化"
            offset, _ = self._intermediate_slice(0 if proj == "gate_proj" else 1, unit)
            # _intermediate_slice 返回的 offset 已经是 unit(元素 / 128 块)为单位
            dst = param.data[expert, offset:offset + local]
            src = loaded_weight.narrow(0, start, local)
        assert dst.shape == src.shape, (
            f"专家权重形状不匹配:{proj}.{kind} 期望 {tuple(dst.shape)},"
            f"checkpoint {tuple(src.shape)}")
        dst.copy_(src)
        return attr

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        out = self.quant_method.apply(self, hidden_states, topk_ids, topk_weights)
        out = out.to(hidden_states.dtype)
        if self.tp_size > 1:
            # 各 rank 只算了自己那片中间维,专家输出是线性的,一次 reduce 即可
            dist.all_reduce(out)
        return out


class Qwen3_5MoeSparseMoeBlock(nn.Module):
    """稀疏 MoE FFN:top-k 专家 + 带 sigmoid 门控的 shared expert。"""

    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        self.gate = Qwen3_5MoeTopKRouter(config)
        self.experts = Qwen3_5MoeExperts(config, quant_config, f"{prefix}.experts")
        self.shared_expert = Qwen3_5MLP(
            config, config.shared_expert_intermediate_size,
            quant_config, f"{prefix}.shared_expert")
        # 每 token 一个标量门(sigmoid),checkpoint 里是 [1, H] 的 bf16 权重
        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size, 1, bias=False, prefix=f"{prefix}.shared_expert_gate")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        topk_weights, topk_ids = self.gate(hidden_states)
        out = self.experts(hidden_states, topk_ids, topk_weights)
        shared = self.shared_expert(hidden_states)
        shared = shared * torch.sigmoid(self.shared_expert_gate(hidden_states))
        return out + shared


class Qwen3_5MoeDecoderLayer(nn.Module):
    """和 dense 版唯一的差别:mlp 换成稀疏 MoE(每层都是 MoE,没有 dense 层)。"""

    def __init__(self, config, layer_idx: int, quant_config=None, prefix: str = ""):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(
                config, layer_idx, quant_config, f"{prefix}.linear_attn")
        elif self.block_type == "full_attention":
            self.self_attn = Qwen3_5Attention(
                config, layer_idx, quant_config, f"{prefix}.self_attn")
        else:
            raise ValueError(f"未知的 layer_types[{layer_idx}]={self.block_type!r}")
        self.mlp = Qwen3_5MoeSparseMoeBlock(config, quant_config, f"{prefix}.mlp")
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.block_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        else:
            hidden_states = self.self_attn(hidden_states, position_ids=position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3_5MoeTextModel(Qwen3_5TextModel):
    """只换掉 decoder layer 类型,forward / 线性状态槽位逻辑全部继承。"""

    def __init__(self, config, quant_config=None, prefix: str = "model.language_model"):
        nn.Module.__init__(self)
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3_5MoeDecoderLayer(config, layer_idx, quant_config,
                                    f"{prefix}.layers.{layer_idx}")
             for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.has_linear_attention = any(
            t == "linear_attention" for t in config.layer_types)
        self.linear_padding_slot: int | None = None


class Qwen3_5MoeModel(Qwen3_5Model):
    """视觉塔 + MoE 语言模型,多模态与 MRoPE 逻辑全部继承 dense 版。"""

    def __init__(self, config, quant_config=None, prefix: str = "model"):
        nn.Module.__init__(self)
        self.config = config
        self.visual = Qwen3_5VisionModel(config.vision_config)
        self.language_model = Qwen3_5MoeTextModel(
            config.text_config, quant_config, f"{prefix}.language_model")
        self.rope_deltas = None


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM):
    """对外入口;只有构造和 per-expert 权重加载是自己实现,其余全部继承。"""

    # checkpoint 把专家存成 experts.<e>.{gate,up,down}_proj.{weight,weight_scale_inv}
    _EXPERT_KEY = re.compile(
        r"^(?P<base>.*\.mlp\.experts)\.(?P<expert>\d+)\."
        r"(?P<proj>gate_proj|up_proj|down_proj)\."
        r"(?P<kind>weight|weight_scale_inv)$")

    def __init__(self, config):
        nn.Module.__init__(self)
        self.model = Qwen3_5MoeModel(config, getattr(config, "quant_config", None))
        self.lm_head = ParallelLMHead(
            config.text_config.vocab_size, config.text_config.hidden_size)
        if config.text_config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.language_model.embed_tokens.weight.data

    def resolve_stacked_weight(self, weight_name: str, loaded_weight: torch.Tensor):
        """把 per-expert 的 2D 权重灌进堆叠的 3D 专家参数。

        返回被写入的参数全名(供 AutoModelForCausalLM 记录),不匹配时返回
        None 交回默认加载路径(融合 3D checkpoint 就走默认路径)。
        """
        match = self._EXPERT_KEY.match(weight_name)
        if match is None:
            return None
        base = match.group("base")
        experts = self.get_submodule(base)
        suffix = experts.load_expert_tensor(
            match.group("proj"), match.group("kind"),
            match.group("expert"), loaded_weight)
        return f"{base}.{suffix}"
