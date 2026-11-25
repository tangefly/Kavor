import torch
import torch.distributed as dist
from torch import nn

from kavor.layers.quant.base import BlockQuantScaleParameter, UnquantizedLinearMethod


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """Base class for TP linear layers: owns the parallel context and quant method."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        if quant_config is None:
            self.quant_method = UnquantizedLinearMethod()
        else:
            self.quant_method = quant_config.get_linear_method(prefix)
        self.quant_method.create_weights(self, input_size, output_size, self.weight_loader)
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """Not sharded: every TP rank keeps the full weight and computes independently."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__(input_size, output_size, bias, quant_config=quant_config, prefix=prefix)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        assert param.data.shape == loaded_weight.shape
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_method.apply(self, x, self.bias)


class ColumnParallelLinear(LinearBase):
    """Sharded on the output dim (dim 0): each rank owns ``output_size / tp_size`` rows."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config=None,
        prefix: str = "",
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0,
                         quant_config=quant_config, prefix=prefix)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_method.apply(self, x, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Sharded on dim 0 over concatenated matrices; ``loaded_shard_id`` picks the block to load."""

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        quant_config=None,
        prefix: str = "",
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias,
                         quant_config=quant_config, prefix=prefix)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        if isinstance(param, BlockQuantScaleParameter):
            block_n = self.weight_block_size[0]
            shard_offset, shard_size = shard_offset // block_n, shard_size // block_n
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """Sharded on dim 0 by heads: Q heads and KV heads are each split by ``tp_size``."""

    def __init__(
        self,
        output_size: int,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        q_gate: bool = False,
        bias: bool = False,
        quant_config=None,
        prefix: str = "",
    ):
        tp_size = dist.get_world_size()
        # With gated attention, q_proj packs q and its gate together, so the q
        # block is twice as wide; weight_loader needs this to offset k/v.
        self.q_mult = 2 if q_gate else 1
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        super().__init__(hidden_size, output_size, bias,
                         quant_config=quant_config, prefix=prefix)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        q_size  = self.q_mult * self.num_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size
        if loaded_shard_id == "q":
            shard_size = q_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = kv_size
            shard_offset = q_size
        else:
            shard_size = kv_size
            shard_offset = q_size + kv_size
        if isinstance(param, BlockQuantScaleParameter):
            block_n = self.weight_block_size[0]
            shard_size, shard_offset = shard_size // block_n, shard_offset // block_n
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """Sharded on the input dim (dim 1); partial outputs are summed with ``all_reduce``."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config=None,
        prefix: str = "",
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1,
                         quant_config=quant_config, prefix=prefix)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.quant_method.apply(self, x, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
