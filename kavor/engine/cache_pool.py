import math
from dataclasses import dataclass

import torch
import torch.distributed as dist

from kavor.engine.sequence import Sequence
from kavor.utils.logger import logger


@dataclass(frozen=True)
class CacheLayout:
    attn_slots: list[tuple[torch.nn.Module, int]]   # [(attention module, KV slot index)]
    linear_modules: list[torch.nn.Module]           # all linear-attention layers
    num_pools: int                                  # number of pools (= max layers per state group)
    num_blocks: int                                 # rows per pool (the range of block ids)
    block_size: int                                 # tokens per KV block
    page_bytes: int                                 # bytes per page per layer
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    dtype_size: int
    num_state_groups: int                           # number of state block groups for linear layers
    max_state_bytes: int                            # bytes used by the heaviest linear state


class CachePool:

    def __init__(self, model: torch.nn.Module, config, world_size: int):
        self.model = model
        self.config = config
        self.world_size = world_size
        self.layout: CacheLayout | None = None
        self.pools: torch.Tensor | None = None
        self.linear_state_len: list[int] = []

    # plan and allocate

    def allocate(self) -> CacheLayout:
        """Plan the layout, allocate the page pool, and bind it to the model.

        Returns the layout so the caller can publish its sizes.
        """
        layout = self._plan_layout()
        self.layout = layout
        self.pools = torch.empty(
            layout.num_pools, layout.num_blocks, layout.page_bytes, dtype=torch.uint8)

        self._bind_attention_pages(layout)
        self._bind_linear_state_pages(layout)
        self._init_linear_state(layout)
        return layout

    @property
    def block_size(self) -> int:
        assert self.layout is not None, "page pool has not been allocated yet"
        return self.layout.block_size

    @property
    def num_blocks(self) -> int:
        assert self.layout is not None, "page pool has not been allocated yet"
        return self.layout.num_blocks

    @property
    def num_state_groups(self) -> int:
        assert self.layout is not None, "page pool has not been allocated yet"
        return self.layout.num_state_groups

    @property
    def has_linear_attention(self) -> bool:
        assert self.layout is not None, "page pool has not been allocated yet"
        return bool(self.layout.linear_modules)

    @property
    def linear_padding_slot(self) -> int | None:
        """Padding row for CUDA graphs (the last row of the pool); None without linear layers."""
        return self.num_blocks - 1 if self.has_linear_attention else None

    # meta data

    def prepare_linear_slot_ids(self, seqs: list[Sequence]) -> torch.Tensor | None:
        """State block ids for this batch, shape [B, num_state_groups] (None without linear layers).

        Like block_table, this is batch metadata: the model only uses it for indexing,
        with no dict lookups and no GPU->CPU sync.
        """
        if self.linear_padding_slot is None:
            return None
        rows = []
        for seq in seqs:
            ids = seq.state_block_ids
            if ids is None:
                # Warmup builds fake sequences without the scheduler, so point them at the padding block
                assert not seq.block_table, (
                    f"Sequence {seq.seq_id} is about to run without state blocks: "
                    "they are allocated during prefill scheduling")
                rows.append([self.linear_padding_slot] * self.num_state_groups)
                continue
            assert len(ids) == self.num_state_groups, (
                f"Sequence {seq.seq_id} has {len(ids)} state blocks, "
                f"but the model needs {self.num_state_groups}")
            assert all(0 <= i < self.linear_padding_slot for i in ids), (
                f"State block ids out of range ({ids}): valid range is [0, {self.linear_padding_slot}); "
                "the last block is reserved for CUDA graphs")
            assert len(set(ids)) == len(ids), (
                f"Duplicate state blocks in one sequence ({ids}): "
                "block groups sharing a page would overwrite each other")
            rows.append(ids)
        return torch.tensor(rows, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)

    def check_linear_states(self, seqs: list[Sequence]):
        """Self-check before decode: each sequence's state blocks should stop exactly at len(seq) - 1 tokens.

        Block ids come from BlockManager; reuse or misalignment fails loudly here instead of
        silently producing garbage.
        """
        if self.linear_padding_slot is None:
            return
        for seq in seqs:
            assert seq.state_block_ids is not None, (
                f"Sequence {seq.seq_id} is about to decode without state blocks")
            got, want = self.linear_state_len[seq.state_block_ids[0]], len(seq) - 1
            assert got == want, (
                f"Sequence {seq.seq_id} has a misaligned state: blocks {seq.state_block_ids} "
                f"stopped at token {got}, but this step handles token {want} "
                "(did prefill not run? did a prefix-cache hit skip the prefix? "
                "was it preempted without re-prefill?)")

    def record_linear_states(self, seqs: list[Sequence]):
        """Record how many tokens each sequence's state blocks have consumed.

        Pure host-side bookkeeping: no GPU work and no synchronization.
        """
        if self.linear_padding_slot is None:
            return
        for seq in seqs:
            if seq.state_block_ids is None:
                continue
            for block_id in seq.state_block_ids:
                self.linear_state_len[block_id] = seq.num_cached_tokens + seq.num_scheduled_tokens

    # layout planning

    @staticmethod
    def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
        strides = []
        acc = 1
        for size in reversed(shape):
            strides.append(acc)
            acc *= size
        return tuple(reversed(strides))

    @staticmethod
    def _linear_state_nbytes(module) -> int:
        """Bytes a linear-attention layer's state page needs (conv_state first, recurrent_state after)."""
        conv_shape, rec_shape = module.state_shapes
        conv_dtype, rec_dtype = module.state_dtypes
        conv_bytes = math.prod(conv_shape) * conv_dtype.itemsize
        rec_offset = (conv_bytes + rec_dtype.itemsize - 1) // rec_dtype.itemsize
        return rec_offset * rec_dtype.itemsize + math.prod(rec_shape) * rec_dtype.itemsize

    def _all_reduce_int(self, value: int, op) -> int:
        """All ranks must agree on block_size/num_blocks, otherwise the block tables will not line up."""
        if self.world_size == 1:
            return value
        t = torch.tensor([value], dtype=torch.int64, device="cuda")
        dist.all_reduce(t, op=op)
        return int(t.item())

    def _collect_attention_slots(self) -> list[tuple[torch.nn.Module, int]]:
        """List every attention KV slot in model order; each slot will own one layer page pool."""
        slots = []
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                slots.extend((module, i) for i in range(module.num_kv_slots))
        assert slots, "Pure linear-attention models have no page-pool layout yet; not supported"
        return slots

    def _collect_linear_layers(self) -> list[torch.nn.Module]:
        return [m for m in self.model.modules() if hasattr(m, "state_shapes")]

    def _choose_block_size(self, max_state_bytes: int, attn_bytes_per_token: int) -> int:
        """Increase the attention logical block_size so one attention page can hold one linear state.

        Any shortfall is padded on the state page (see _bind_state_page).
        FlashAttention's paged KV requires page_block_size to be a multiple of 256.
        """
        block_size = self.config.kvcache_block_size
        if max_state_bytes == 0:
            return block_size
        need = (max_state_bytes + attn_bytes_per_token - 1) // attn_bytes_per_token
        block_size = max(block_size, ((need + 255) // 256) * 256)
        return self._all_reduce_int(block_size, dist.ReduceOp.MAX)

    def _probe_num_blocks(self, num_pools: int, page_bytes: int) -> int:
        """Compute the number of page-pool rows from free GPU memory; each block owns one page per pool."""
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        budget = int(total * self.config.gpu_memory_utilization - used - peak + current)
        num_blocks = self._all_reduce_int(
            budget // (num_pools * page_bytes), dist.ReduceOp.MIN)
        assert num_blocks >= 2, (
            f"Page pool has only {num_blocks} blocks: at least 1 is needed for requests "
            "and 1 for CUDA graph padding. Increase gpu_memory_utilization or reduce max_model_len")
        return num_blocks

    def _plan_layout(self) -> CacheLayout:
        text_config = self.config.text_config
        num_kv_heads = text_config.num_key_value_heads // self.world_size
        head_dim = getattr(
            text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads)
        dtype_size = text_config.dtype.itemsize

        attn_slots = self._collect_attention_slots()
        linear_modules = self._collect_linear_layers()
        max_state_bytes = max(
            (self._linear_state_nbytes(m) for m in linear_modules), default=0)

        attn_bytes_per_token = 2 * num_kv_heads * head_dim * dtype_size
        block_size = self._choose_block_size(max_state_bytes, attn_bytes_per_token)

        page_bytes = attn_bytes_per_token * block_size
        assert max_state_bytes <= page_bytes, (
            f"Linear state ({max_state_bytes} B) does not fit in one attention page ({page_bytes} B)")

        num_pools = len(attn_slots)
        num_state_groups = ((len(linear_modules) + num_pools - 1) // num_pools
                            if linear_modules else 0)
        num_blocks = self._probe_num_blocks(num_pools, page_bytes)

        return CacheLayout(
            attn_slots=attn_slots,
            linear_modules=linear_modules,
            num_pools=num_pools,
            num_blocks=num_blocks,
            block_size=block_size,
            page_bytes=page_bytes,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=text_config.dtype,
            dtype_size=dtype_size,
            num_state_groups=num_state_groups,
            max_state_bytes=max_state_bytes,
        )

    # memory binding

    def _bind_attention_pages(self, layout: CacheLayout):
        """Attention slot i maps to pool i; K lives in the first half of each page, V in the second half."""
        block_elems = layout.block_size * layout.num_kv_heads * layout.head_dim
        block_stride = layout.page_bytes // layout.dtype_size
        inner = (layout.num_kv_heads * layout.head_dim, layout.head_dim, 1)
        shape = (layout.num_blocks, layout.block_size, layout.num_kv_heads, layout.head_dim)
        for pool, (module, slot) in enumerate(layout.attn_slots):
            flat = self.pools[pool].view(layout.dtype)
            base = flat.storage_offset()
            module.k_cache[slot] = flat.as_strided(shape, (block_stride, *inner), base)
            module.v_cache[slot] = flat.as_strided(
                shape, (block_stride, *inner), base + block_elems)

    def _bind_linear_state_pages(self, layout: CacheLayout):
        """Linear layer j maps to pool j % num_pools; layers in the same group share one state block id."""
        for j, module in enumerate(layout.linear_modules):
            module.state_group = j // layout.num_pools
            self._bind_state_page(module, self.pools[j % layout.num_pools], layout)

    def _bind_state_page(self, module, pool: torch.Tensor, layout: CacheLayout):
        conv_shape, rec_shape = module.state_shapes
        conv_dtype, rec_dtype = module.state_dtypes
        conv_bytes = math.prod(conv_shape) * conv_dtype.itemsize
        rec_offset = (conv_bytes + rec_dtype.itemsize - 1) // rec_dtype.itemsize
        state_bytes = rec_offset * rec_dtype.itemsize + math.prod(rec_shape) * rec_dtype.itemsize
        assert state_bytes <= layout.page_bytes, (
            f"Linear state ({state_bytes} B) does not fit in one page ({layout.page_bytes} B)")
        assert (layout.page_bytes % conv_dtype.itemsize == 0
                and layout.page_bytes % rec_dtype.itemsize == 0)

        conv_flat = pool.view(conv_dtype)
        rec_flat = pool.view(rec_dtype)
        module.conv_pool = conv_flat.as_strided(
            (layout.num_blocks, *conv_shape),
            (layout.page_bytes // conv_dtype.itemsize, *self._contiguous_strides(conv_shape)),
            conv_flat.storage_offset())
        module.recurrent_pool = rec_flat.as_strided(
            (layout.num_blocks, *rec_shape),
            (layout.page_bytes // rec_dtype.itemsize, *self._contiguous_strides(rec_shape)),
            rec_flat.storage_offset() + rec_offset)

    def _init_linear_state(self, layout: CacheLayout):
        """Initialize the state ledger; zero the padding block so CUDA graph padding rows are harmless."""
        self.linear_state_len = [-1] * layout.num_blocks
        if not layout.linear_modules:
            return
        self.pools[:, self.linear_padding_slot].zero_()

    def log_summary(self):
        layout = self.layout
        assert layout is not None, "page pool has not been allocated yet"
        logger.info(
            "统一页池: %d 块 × %d 层页 × %.2f MB = %.2f GB (block_size=%d, 每 token %.1f KB)",
            layout.num_blocks, layout.num_pools, layout.page_bytes / 2**20,
            layout.num_blocks * layout.num_pools * layout.page_bytes / 2**30,
            layout.block_size, layout.page_bytes / layout.block_size / 1024,
        )
        if layout.linear_modules:
            logger.info(
                "线性注意力状态: 每层 %.2f MB × %d 层, 分 %d 个块组(每组 %d 层);"
                " 状态页 padding %.2f MB/层",
                layout.max_state_bytes / 2**20, len(layout.linear_modules),
                layout.num_state_groups, layout.num_pools,
                (layout.page_bytes - layout.max_state_bytes) / 2**20,
            )
