from .qwen3 import Qwen3Paser
from .registry import register_parser


@register_parser("qwen3_5")
class Qwen3_5Paser(Qwen3Paser):
    pass
