from collections import OrderedDict

from .auto_factory import _LazyTypeMapping


MODEL_MAPPING_NAMES = OrderedDict(
    [
        ("qwen3", "Qwen3ForCausalLM"),
        ("ernie4_5", "Ernie4_5ForCausalLM"),
        ("ouro", "OuroForCausalLM"),
        ("qwen3_5", "Qwen3_5ForCausalLM"),
        ("qwen3_vl", "Qwen3VLForCausalLM"),
        ("mistral", "MistralForCausalLM"),
    ]
)

MODEL_FOR_CAUSAL_LM_MAPPING = _LazyTypeMapping(MODEL_MAPPING_NAMES)
