from collections import OrderedDict

from .auto_factory import _LazyTypeMapping


MODEL_MAPPING_NAMES = OrderedDict(
    [
        # Model for Causal LM mapping
        ("qwen3", "Qwen3ForCausalLM"),
        ("qwen3_5", "Qwen3_5ForCausalLM"),  # VLM compatibility
    ]
)

MODEL_FOR_CAUSAL_LM_MAPPING = _LazyTypeMapping(MODEL_MAPPING_NAMES)
