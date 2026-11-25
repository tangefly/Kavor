from .qwen3_5 import Qwen3_5Paser
from .registry import register_parser


@register_parser("qwen3_5_moe")
class Qwen3_5MoePaser(Qwen3_5Paser):
    pass
