from glob import glob
import os
from safetensors import safe_open
from torch import nn
from tqdm import tqdm
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
    def _assign_weights(
        cls: type["AutoModelForCausalLM"],
        model: nn.Module,
        weights,
        packed_modules_mapping: dict,
    ) -> set[str]:
        loaded = set()
        for weight_name, loaded_weight in weights:
            for k in packed_modules_mapping:
                if k in weight_name:
                    v, shard_id = packed_modules_mapping[k]
                    param_name = weight_name.replace(k, v)
                    param = model.get_parameter(param_name)
                    getattr(param, "weight_loader")(param, loaded_weight, shard_id)
                    loaded.add(param_name)
                    break
            else:
                param = model.get_parameter(weight_name)
                weight_loader = getattr(param, "weight_loader", cls.default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded.add(weight_name)
        return loaded

    @classmethod
    def _load_safetensors(
        cls: type["AutoModelForCausalLM"],
        model: nn.Module,
        file: str,
        packed_modules_mapping: dict,
    ) -> set[str]:
        with safe_open(file, "pt", "cpu") as f:
            return cls._assign_weights(
                model, ((name, f.get_tensor(name)) for name in f.keys()), packed_modules_mapping)

    @classmethod
    def _load_bin(
        cls: type["AutoModelForCausalLM"],
        model: nn.Module,
        file: str,
        packed_modules_mapping: dict,
    ) -> set[str]:
        state_dict = torch.load(file, map_location="cpu", weights_only=True)
        return cls._assign_weights(model, state_dict.items(), packed_modules_mapping)

    @classmethod
    def load_weight(
        cls: type["AutoModelForCausalLM"],
        model: nn.Module,
        path: str
    ):
        packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

        files = sorted(glob(os.path.join(path, "*.safetensors")))
        if files:
            load_file = cls._load_safetensors
        else:
            files = sorted(f for f in glob(os.path.join(path, "*.bin"))
                           if os.path.basename(f) != "training_args.bin")
            load_file = cls._load_bin
        if not files:
            raise FileNotFoundError(
                f"{path} 下没有找到权重文件(*.safetensors 或 *.bin),"
                f"确认 --model 指向的是完整的 HF 模型目录")

        loaded: set[str] = set()
        for file in tqdm(files, desc="Loading weights"):
            loaded |= load_file(model, file, packed_modules_mapping)
        cls._check_all_loaded(model, loaded)

    @classmethod
    def _check_all_loaded(
        cls: type["AutoModelForCausalLM"],
        model: nn.Module,
        loaded: set[str],
    ):
        covered = {p.data_ptr() for name, p in model.named_parameters() if name in loaded}
        missing = [name for name, p in model.named_parameters()
                   if name not in loaded and p.data_ptr() not in covered]
        if missing:
            preview = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
            raise RuntimeError(
                f"有 {len(missing)} 个参数没有从 checkpoint 加载到: {preview}")
