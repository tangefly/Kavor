
from collections import OrderedDict
import importlib
from typing import Any

class _LazyTypeMapping(OrderedDict):
    """
    A dictionary that lazily loads model classes when they are requested.
    """

    def __init__(self, mapping) -> None:
        self._mapping = mapping
        self._modules = {}

    def __getitem__(self, key: str):
        if key not in self._modules:
            class_name = self._mapping[key]
            module = importlib.import_module(f".{key}", "nanovllm.models")
            self._modules[key] = getattr(module, class_name)
        return self._modules[key]