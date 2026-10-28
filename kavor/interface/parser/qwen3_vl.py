from .qwen3 import Qwen3Paser
from .registry import register_parser


@register_parser("qwen3_vl")
class Qwen3VLPaser(Qwen3Paser):
    pass