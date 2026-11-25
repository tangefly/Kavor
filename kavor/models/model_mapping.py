from collections import OrderedDict

from .auto_factory import _LazyTypeMapping

MODEL_MAPPING_NAMES = OrderedDict(
    [
        ("ernie4_5", "Ernie4_5ForCausalLM"),
        ("mistral", "MistralForCausalLM"),
        ("ouro", "OuroForCausalLM"),
        ("qwen3", "Qwen3ForCausalLM"),
        ("qwen3_vl", "Qwen3VLForCausalLM"),
        ("qwen3_5", "Qwen3_5ForCausalLM"),
        ("qwen3_5_moe", "Qwen3_5MoeForCausalLM")
    ]
)

MODEL_FOR_CAUSAL_LM_MAPPING = _LazyTypeMapping(MODEL_MAPPING_NAMES)
