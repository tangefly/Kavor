import importlib
import json
import os
from inspect import signature

import torch

CONFIG_REGISTRY: dict[str, type] = {}

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


class KavorConfig:

    model_type: str = ""

    def __init__(self, **kwargs):
        self.extra = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._raw: dict = {}
        self.name_or_path: str = ""
        self.dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_json(cls, model_dir: str, config_dict: dict) -> "KavorConfig":
        params = signature(cls.__init__).parameters
        accepts_extra = any(p.kind is p.VAR_KEYWORD for p in params.values())
        kwargs = config_dict if accepts_extra else {
            k: v for k, v in config_dict.items() if k in params
        }
        config = cls(**kwargs)

        config.name_or_path = model_dir
        config.dtype = resolve_dtype(
            config_dict.get("torch_dtype") or config_dict.get("dtype"))
        config._raw = config_dict

        text_config = config_dict.get("text_config")
        if isinstance(text_config, dict):
            config.text_config = cls.from_json(model_dir, text_config)

        return config


class AutoConfig:

    @classmethod
    def from_pretrained(cls, model_dir: str, **kwargs):
        config_dict = read_model_config(model_dir)
        model_type = config_dict.get("model_type")

        config_cls = _resolve_config_cls(model_type)
        if config_cls is not None:
            return config_cls.from_json(model_dir, config_dict)

        from transformers import AutoConfig as HFAutoConfig
        try:
            return HFAutoConfig.from_pretrained(model_dir, trust_remote_code=False, **kwargs)
        except Exception as e:
            raise ValueError(
                f"model_type {model_type!r} is neither registered in Kavor nor "
                f"implemented natively by transformers. Please write a KavorConfig "
                f"subclass in kavor/models/{model_type}.py and annotate it with "
                f"@register_config({model_type!r})"
            ) from e


def register_config(model_type: str):

    def decorator(cls):
        cls.model_type = model_type
        CONFIG_REGISTRY[model_type] = cls
        return cls

    return decorator


def read_model_config(model_dir: str) -> dict:
    path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no config.json found under {model_dir}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_model_type(model_dir: str) -> str:
    """Read only model_type, without building a config object (for the parser registry)."""
    return read_model_config(model_dir).get("model_type")


def resolve_dtype(value) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if value is None or value == "auto":
        return torch.bfloat16
    dtype = _DTYPE_MAP.get(str(value).removeprefix("torch."))
    if dtype is None:
        raise ValueError(f"unknown dtype: {value!r}; please add it to _DTYPE_MAP")
    return dtype


def _resolve_config_cls(model_type: str) -> type | None:
    if model_type in CONFIG_REGISTRY:
        return CONFIG_REGISTRY[model_type]
    try:
        # Convention: model_type is the module filename; importing it triggers @register_config
        importlib.import_module(f"kavor.models.{model_type}")
    except ImportError:
        pass
    return CONFIG_REGISTRY.get(model_type)
