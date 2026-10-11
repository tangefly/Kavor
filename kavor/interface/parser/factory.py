from transformers import AutoConfig

from .registry import PARSER_REGISTRY


def get_paser(model: str):
    config = AutoConfig.from_pretrained(model)
    model_type = config.model_type
    
    parser_cls = PARSER_REGISTRY.get(
        model_type
    )
    
    if parser_cls is None:
        raise ValueError(
            f"No parser found for model type: {model_type}"
        )

    return parser_cls()