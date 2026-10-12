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


def register_config(model_type: str):

    def decorator(cls):
        cls.model_type = model_type
        CONFIG_REGISTRY[model_type] = cls
        return cls

    return decorator


def read_config_json(model_dir: str) -> dict:
    path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{model_dir} 下没有 config.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_model_type(model_dir: str) -> str:
    """只取 model_type,不构造 config 对象(parser 注册表用)。"""
    return read_config_json(model_dir).get("model_type")


def resolve_dtype(value) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if value is None or value == "auto":
        return torch.bfloat16
    dtype = _DTYPE_MAP.get(str(value).removeprefix("torch."))
    if dtype is None:
        raise ValueError(f"不认识的 dtype: {value!r},请在 _DTYPE_MAP 里补上")
    return dtype


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


def _resolve_config_cls(model_type: str) -> type | None:
    if model_type in CONFIG_REGISTRY:
        return CONFIG_REGISTRY[model_type]
    try:
        # 约定:model_type 就是文件名,导入模块即触发 @register_config
        importlib.import_module(f"kavor.models.{model_type}")
    except ImportError:
        pass
    return CONFIG_REGISTRY.get(model_type)


class AutoConfig:

    @classmethod
    def from_pretrained(cls, model_dir: str, **kwargs):
        config_dict = read_config_json(model_dir)
        model_type = config_dict.get("model_type")

        config_cls = _resolve_config_cls(model_type)
        if config_cls is not None:
            return config_cls.from_json(model_dir, config_dict)

        # 没注册过的类型(qwen3/ernie4_5/mistral 等 transformers 内置架构)交给 HF。
        # 固定 trust_remote_code=False:只认 transformers 自带的本地实现,
        # 既不会弹交互提示,也不会执行模型仓库里的代码。
        from transformers import AutoConfig as HFAutoConfig
        try:
            return HFAutoConfig.from_pretrained(model_dir, trust_remote_code=False, **kwargs)
        except Exception as e:
            raise ValueError(
                f"model_type={model_type!r} 既没有在 Kavor 注册,transformers 也没有内置实现。"
                f"请在 kavor/models/{model_type}.py 里写一个 KavorConfig 子类并打上 "
                f"@register_config({model_type!r})") from e
