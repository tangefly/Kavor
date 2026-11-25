import os
from glob import glob

import torch
from safetensors import safe_open
from torch import nn
from tqdm import tqdm

from kavor.layers.quant.fp8 import Fp8Config
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
        model_class = cls.get_model_class(hfconfig)
        hfconfig.quant_config = Fp8Config.from_config(
            getattr(hfconfig, "quantization_config", None)
        )
        model = model_class(hfconfig)
        cls.load_weight(model, hfconfig.name_or_path, use_tqdm)
        cls._prepare_quantized_layers(model)
        return model

    @classmethod
    def _prepare_quantized_layers(cls, model: nn.Module) -> None:
        """Pick a kernel for every quantized Linear and convert its layout once.

        Runs after the weights are loaded. Must happen before the KV cache is
        allocated and the CUDA graphs are captured (see model_runner).
        """
        # kernel name -> (scheme, layer count); per-layer choices are debug logs,
        # the whole model is summarized in one info line below
        kernels: dict[str, tuple[str, int]] = {}
        experts = 0
        for module in model.modules():
            method = getattr(module, "quant_method", None)
            if method is None or not hasattr(method, "prepare_kernel"):
                continue
            method.prepare_kernel(module)
            kernel = getattr(method, "kernel", None)
            if kernel is None:
                # MoE experts pick their kernel inside fused_moe, not here
                experts += 1
                continue
            scheme, count = kernels.get(kernel.name, (kernel.scheme, 0))
            kernels[kernel.name] = (scheme, count + 1)
        if kernels or experts:
            cls._log_quant_summary(kernels, experts)

    @staticmethod
    def _log_quant_summary(kernels: dict[str, tuple[str, int]], experts: int) -> None:
        def count(n: int, noun: str) -> str:
            return f"{n} {noun}" + ("" if n == 1 else "s")

        by_kernel = sorted(kernels.items(), key=lambda kv: -kv[1][1])
        # First group carries the noun ("407 linear layers on marlin (W8A16)"),
        # later ones drop it ("7 on triton_w8a8 (W8A8)") to stay short.
        groups = [
            f"{count(n, 'linear layer') if i == 0 else n} on {name} ({scheme})"
            for i, (name, (scheme, n)) in enumerate(by_kernel)
        ]
        if experts:
            groups.append(f"{count(experts, 'MoE expert module')} on fused_moe")
        logger.info("FP8 quantization: %s", ", ".join(groups))

    @classmethod
    def get_model_class(
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
        """Copy the checkpoint weights into the model by parameter name.

        Returns (names of loaded parameters, ignored keys with their byte size).
        Which keys are ignored is declared explicitly by the model class via
        ignore_weight_prefixes (e.g. mtp.); without a declaration nothing is ignored.
        """
        ignore_prefixes = getattr(model, "ignore_weight_prefixes", ())
        loaded = set()
        skipped = []
        for weight_name, loaded_weight in weights:
            if weight_name.startswith(ignore_prefixes):
                skipped.append((weight_name, loaded_weight.numel() * loaded_weight.element_size()))
                continue
            # Per-expert weights (the checkpoint stores each expert separately) are
            # packed into a stacked 3D parameter: the model declares a resolver that
            # returns the full name of the parameter it wrote, or None to fall back
            # to the default path
            resolve_stacked = getattr(model, "resolve_stacked_weight", None)
            if resolve_stacked is not None:
                stacked_name = resolve_stacked(weight_name, loaded_weight)
                if stacked_name is not None:
                    loaded.add(stacked_name)
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
                f"No weight files (*.safetensors or *.bin) found under {path}; "
                f"make sure --model points to a complete HF model directory")

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
            "Ignored %d weights as declared (prefixes %s, %.1f MB total): %s",
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
                f"{len(missing)} parameters were not loaded from the checkpoint: {preview}")
