from .qwen3 import Qwen3Paser
from .registry import register_parser


@register_parser("ouro")
class OuroPaser(Qwen3Paser):
    # Ouro 的 chat template 与 Qwen3 同源(<think>/</think>、<tool_call>),
    # 先复用,等标记分叉了再在这里覆盖
    pass
