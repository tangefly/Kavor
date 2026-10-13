from kavor.models.configuration_auto import read_model_type

from .registry import PARSER_REGISTRY


def get_paser(model: str, tokenizer=None):
    """按 model_type 取 parser;tokenizer 用来收集需要清理的特殊 token。"""
    model_type = read_model_type(model)

    parser_cls = PARSER_REGISTRY.get(
        model_type
    )

    if parser_cls is None:
        raise ValueError(
            f"No parser found for model type: {model_type}"
        )

    return parser_cls(tokenizer)
