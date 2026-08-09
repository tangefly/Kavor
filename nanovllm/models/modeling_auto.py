from glob import glob
import os
from safetensors import safe_open
from torch import nn
import torch

from .model_mapping import MODEL_FOR_CAUSAL_LM_MAPPING

class AutoModelForCausalLM:
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING
    
    def __init__(self):
        pass
    
    @classmethod
    def from_pretrained(
        cls: type["AutoModelForCausalLM"],
        hfconfig,
        **kwargs,
    ):
        model_class = cls._get_model_class(hfconfig)
        model = model_class(hfconfig)
        cls.load_weight(model, hfconfig.name_or_path)
        return model

    @classmethod
    def _get_model_class(
        cls: type["AutoModelForCausalLM"],
        hfconfig,
        **kwargs,
    ):
        model_type = hfconfig.model_type
        return cls._model_mapping[model_type]
    
    @classmethod         
    def default_weight_loader(
        cls: type["AutoModelForCausalLM"], 
        param: nn.Parameter, 
        loaded_weight: torch.Tensor
    ):
        param.data.copy_(loaded_weight)

    @classmethod
    def load_weight(
        cls: type["AutoModelForCausalLM"], 
        model: nn.Module, 
        path: str
    ):
        packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
        for file in glob(os.path.join(path, "*.safetensors")):
            with safe_open(file, "pt", "cpu") as f:
                for weight_name in f.keys():
                    for k in packed_modules_mapping:
                        if k in weight_name:
                            v, shard_id = packed_modules_mapping[k]
                            param_name = weight_name.replace(k, v)
                            param = model.get_parameter(param_name)
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(param, f.get_tensor(weight_name), shard_id)
                            break
                    else:
                        param = model.get_parameter(weight_name)
                        weight_loader = getattr(param, "weight_loader", cls.default_weight_loader)
                        weight_loader(param, f.get_tensor(weight_name))
        