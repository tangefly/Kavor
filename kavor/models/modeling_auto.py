from glob import glob
import os
from safetensors import safe_open
from torch import nn
from tqdm import tqdm
import torch

from kavor.utils.logger import logger

from .model_mapping import MODEL_FOR_CAUSAL_LM_MAPPING


class AutoModelForCausalLM:
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING

    def __init__(self):
        pass

    @classmethod
    def from_pretrained(
        cls: type["AutoModelForCausalLM"],
        hfconfig,
        use_tqdm,
        **kwargs,
    ):
        model_class = cls._get_model_class(hfconfig)
        model = model_class(hfconfig)
        cls.load_weight(model, hfconfig.name_or_path, use_tqdm)
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
    ) -> tuple[set[str], list[tuple[str, int]]]:
        """把 checkpoint 的权重按名字灌进模型。

        返回 (已加载的参数名, 被忽略的 key 及其字节数)。
        忽略哪些由模型类的 ignore_weight_prefixes 显式声明(如 mtp.),没声明就不忽略任何东西。
        """
        ignore_prefixes = getattr(model, "ignore_weight_prefixes", ())
        loaded = set()
        skipped = []
        for weight_name, loaded_weight in weights:
            if weight_name.startswith(ignore_prefixes):
                skipped.append((weight_name, loaded_weight.numel() * loaded_weight.element_size()))
                continue
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
        return loaded, skipped

    @classmethod
    def _load_safetensors(
        cls: type["AutoModelForCausalLM"],
        model: nn.Module,
        file: str,
        packed_modules_mapping: dict,
    ) -> tuple[set[str], list[tuple[str, int]]]:
        with safe_open(file, "pt", "cpu") as f:
            return cls._assign_weights(
                model, ((name, f.get_tensor(name)) for name in f.keys()), packed_modules_mapping)

    @classmethod
    def _load_bin(
        cls: type["AutoModelForCausalLM"],
        model: nn.Module,
        file: str,
        packed_modules_mapping: dict,
    ) -> tuple[set[str], list[tuple[str, int]]]:
        state_dict = torch.load(file, map_location="cpu", weights_only=True)
        return cls._assign_weights(model, state_dict.items(), packed_modules_mapping)

    @classmethod
    def load_weight(
        cls: type["AutoModelForCausalLM"],
        model: nn.Module,
        path: str,
        use_tqdm: bool,
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
        skipped: list[tuple[str, int]] = []
        for file in (tqdm(files, desc="Loading weights") if use_tqdm else files):
            loaded_in_file, skipped_in_file = load_file(model, file, packed_modules_mapping)
            loaded |= loaded_in_file
            skipped.extend(skipped_in_file)
        cls._log_skipped(model, skipped)
        cls._check_all_loaded(model, loaded)

    @classmethod
    def _log_skipped(
        cls: type["AutoModelForCausalLM"],
        model: nn.Module,
        skipped: list[tuple[str, int]],
    ):
        if not skipped:
            return
        ignore_prefixes = getattr(model, "ignore_weight_prefixes", ())
        total_mb = sum(size for _, size in skipped) / 2**20
        prefixes = ", ".join(p for p in ignore_prefixes if any(n.startswith(p) for n, _ in skipped))
        preview = ", ".join(name for name, _ in skipped[:3]) + (", ..." if len(skipped) > 3 else "")
        logger.info(
            "按声明忽略 %d 个权重(前缀 %s, 共 %.1f MB): %s",
            len(skipped), prefixes, total_mb, preview,
        )

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
