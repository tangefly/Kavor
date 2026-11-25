import torch
import torch.distributed as dist
from torch import nn

from kavor.layers.quant.base import BlockQuantScaleParameter, UnquantizedLinearMethod


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

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
        # scale 参数在 dim 0 上就是 block 数,均分逻辑与权重一致
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_method.apply(self, x, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

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
            # scale 的行数是 block 数:元素单位的分片换算成 block 单位
            block_n = self.weight_block_size[0]
            shard_offset, shard_size = shard_offset // block_n, shard_size // block_n
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

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
            # scale 的行数是 block 数:元素单位的分片换算成 block 单位
            block_n = self.weight_block_size[0]
            shard_size, shard_offset = shard_size // block_n, shard_offset // block_n
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

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
        # scale 参数在 dim 1 上就是 block 数,均分逻辑与权重一致
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
