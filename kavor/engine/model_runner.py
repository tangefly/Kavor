import pickle

import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from tqdm.auto import tqdm

from kavor.config import Config
from kavor.engine.cache_pool import CachePool
from kavor.engine.sequence import Sequence
from kavor.layers.sampler import Sampler
from kavor.utils.context import set_context, get_context, reset_context
from kavor.utils.logger import logger
from kavor.utils.mm_plan import plan_shards
from kavor.models import AutoModelForCausalLM


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.enable_log = self.rank == 0

        dist.init_process_group(
            "nccl", f"tcp://localhost:{config.rdzv_port}",
            world_size=self.world_size, rank=rank, device_id=torch.device(f"cuda:{rank}"))
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(config.text_config.dtype)
        torch.set_default_device("cuda")
        Sequence.block_size = config.kvcache_block_size
        self.model = AutoModelForCausalLM.from_pretrained(hf_config, use_tqdm=self.enable_log)
        self.sampler = Sampler()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                try:
                    self.shm = SharedMemory(name="kavor")
                    self.shm.close()
                    self.shm.unlink()
                    logger.warning("Removed stale shared memory.")
                except FileNotFoundError:
                    pass
                self.shm = SharedMemory(name="kavor", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="kavor")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        if n + 4 > len(self.shm.buf):
            raise RuntimeError(
                f"要发给 TP worker 的数据有 {n / 2**20:.1f} MB,超过共享内存 "
                f"{len(self.shm.buf) / 2**20:.0f} MB:args 里不要夹带大张量"
                "(多模态的像素张量应该在 ModalInputs.for_workers 里就被丢掉,"
                "改走 _scatter_mm 那条 NCCL 数据面)")
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def profile_prefill_shape(self) -> tuple[int, int]:
        """Worst-case prefill shape: (longest single sequence, number of sequences)."""
        seq_len = min(self.config.max_num_batched_tokens, self.config.max_model_len)
        num_seqs = max(1, min(self.config.max_num_batched_tokens // seq_len,
                              self.config.max_num_seqs))
        return seq_len, num_seqs

    def _padding_slot_ids(self, bs: int) -> torch.Tensor | None:
        slot = self.cache.linear_padding_slot
        if slot is None:
            return None
        return torch.full((bs, self.cache.num_state_groups), slot,
                          dtype=torch.int64, device="cuda")

    @torch.inference_mode()
    def warmup_model(self):
        """Run the worst-case prefill once: profiling plus the one-off kernel/compile cost.

        The real ``run()`` path cannot be used here: fake sequences have no block table,
        which would leave ``slot_mapping`` empty and trip the store-kvcache assertion.
        """
        seq_len, num_seqs = self.profile_prefill_shape()
        total = seq_len * num_seqs
        input_ids = torch.zeros(total, dtype=torch.int64, device="cuda")
        positions = torch.arange(seq_len, device="cuda").repeat(num_seqs)
        cu_seqlens = torch.arange(0, total + 1, seq_len, dtype=torch.int32, device="cuda")
        # the temporary profiling pool is sized for ``total`` tokens, so these slots are valid
        slot_mapping = torch.arange(total, dtype=torch.int32, device="cuda")
        set_context(True, cu_seqlens, cu_seqlens, seq_len, seq_len,
                    slot_mapping, None, None, self._padding_slot_ids(num_seqs))
        try:
            logits = self.model.compute_logits(self.model(input_ids, positions))
            self.sampler(logits, torch.ones(num_seqs, dtype=torch.float32, device="cuda"))
        finally:
            reset_context()

    @torch.inference_mode()
    def _profile_decode(self, bs: int):
        """Run one decode step at ``bs`` (the largest captured graph) to measure its peak."""
        slot_mapping = torch.zeros(bs, dtype=torch.int32, device="cuda")
        context_lens = torch.ones(bs, dtype=torch.int32, device="cuda")
        block_tables = torch.zeros(bs, 1, dtype=torch.int32, device="cuda")
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens,
                    block_tables=block_tables, linear_slot_ids=self._padding_slot_ids(bs))
        try:
            input_ids = torch.zeros(bs, dtype=torch.int64, device="cuda")
            positions = torch.zeros(bs, dtype=torch.int64, device="cuda")
            logits = self.model.compute_logits(self.model(input_ids, positions))
            self.sampler(logits, torch.ones(bs, dtype=torch.float32, device="cuda"))
        finally:
            reset_context()

    @torch.inference_mode()
    def profile_activation_peak(self, max_bs: int) -> int:
        """Measure prefill/decode activation peaks and return the bytes to keep free.

        With CUDA graphs we must keep a second decode peak: capture warms up eagerly and
        ``torch.cuda.graph`` allocates its own private pool, so the eager warmup's cached
        blocks cannot be reused by the graph pool.

        No extra margin is added on purpose: the pool is sized from
        ``gpu_memory_utilization``, so whatever is left between that and the total stays
        unused and is the actual safety net. Keep the utilization at or below ~0.95 if you
        want that net to stay larger than allocator/graph-exec noise.
        """
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        def measure(fn) -> int:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_stats()["allocated_bytes.all.current"]
            fn()
            torch.cuda.synchronize()
            return torch.cuda.memory_stats()["allocated_bytes.all.peak"] - base

        prefill_peak = measure(self.warmup_model)
        decode_peak = measure(lambda: self._profile_decode(max_bs))
        graph_pool = 0 if self.enforce_eager else decode_peak
        reserve = graph_pool + max(prefill_peak, decode_peak)
        if self.enable_log:
            logger.info(
                "激活 profiling: prefill %.2f GiB, decode(bs=%d) %.2f GiB,"
                " CUDA graph 池 %.2f GiB -> 预留 %.2f GiB",
                prefill_peak / 2**30, max_bs, decode_peak / 2**30,
                graph_pool / 2**30, reserve / 2**30)
        return reserve

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool, mm_kwargs: dict | None = None):
        if is_prefill or self.enforce_eager or input_ids.size(0) > self.graph_bs[-1]:
            logits = self.model(input_ids, positions, **(mm_kwargs or {}))
            return self.model.compute_logits(logits)
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            if graph_vars["linear_slot_ids"] is not None:
                graph_vars["linear_slot_ids"].fill_(self.cache.linear_padding_slot)
                graph_vars["linear_slot_ids"][:bs] = context.linear_slot_ids
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        mm_kwargs = self.prepare_mm_inputs(seqs) if is_prefill else {}
        logits = self.run_model(input_ids, positions, is_prefill, mm_kwargs)
        self.cache.record_linear_states(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        text_config = config.text_config
        max_bs = min(self.config.max_num_seqs, 512, self.cache.max_concurrent())
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, text_config.hidden_size)
        linear_slot_ids = (None if self.cache.linear_padding_slot is None else
                           torch.full((max_bs, self.cache.num_state_groups),
                                      self.cache.linear_padding_slot, dtype=torch.int64))
        self.graph_bs = [bs for bs in (1, 2, 4, 8) if bs <= max_bs]
        self.graph_bs += list(range(16, max_bs + 1, 16))
        if self.graph_bs[-1] != max_bs:
            self.graph_bs.append(max_bs)
        self.graphs = {}
        self.graph_pool = None

        pbar = tqdm(reversed(self.graph_bs), total=len(self.graph_bs),
                    desc="Capturing CUDA graphs", dynamic_ncols=True,
                    disable=not self.enable_log)
        for bs in pbar:
            pbar.set_postfix(batch_size=bs)
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs],
                        block_tables=block_tables[:bs],
                        linear_slot_ids=None if linear_slot_ids is None else linear_slot_ids[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            linear_slot_ids=linear_slot_ids,
            outputs=outputs,
        )

    def allocate_kv_cache(self):
        """两段式建页池:先绑临时小池实测前向峰值,再按预留量规划正式池。

        ``gpu_memory_utilization`` 约束的是「权重 + 页池 + 前向峰值」,而页池大小又反过来
        决定前向峰值(峰值里最大的一项是 batch 维度的线性注意力状态),所以先做一次乐观
        估计拿到 profiling 用的 batch,再实测、再正式规划。
        """
        self.cache = CachePool(self.model, self.config, self.world_size)

        # 1) 乐观规划(不留激活余量):真实容量不会超过它,用它决定 profiling 的 batch 上界
        self.cache.plan()
        probe_bs = min(self.config.max_num_seqs, 512, self.cache.max_concurrent())

        # 2) 绑一个刚好能跑完 prefill profiling 的临时池(prefill 的 slot_mapping 必须落在池内)
        seq_len, num_seqs = self.profile_prefill_shape()
        profile_blocks = (seq_len * num_seqs + self.cache.block_size - 1) // self.cache.block_size + 2
        self.cache.allocate(num_blocks=profile_blocks)

        # 3) 实测峰值,再建正式池(allocate 会先释放临时池)
        reserve = self.profile_activation_peak(probe_bs)
        layout = self.cache.allocate(reserve=reserve)

        self.block_size = layout.block_size
        Sequence.block_size = self.block_size
        self.config.kvcache_block_size = self.block_size
        self.config.num_kvcache_blocks = layout.num_blocks
        self.config.has_linear_attention = self.cache.has_linear_attention
        self.config.num_linear_state_groups = self.cache.num_state_groups

        # 4) 把 max_num_seqs 收敛到页池真正能容纳的并发:Scheduler 和 CUDA graph 都用这个值
        capacity = self.cache.max_concurrent()
        if capacity < self.config.max_num_seqs:
            if self.enable_log:
                logger.warning(
                    "max_num_seqs %d -> %d:页池只有 %d 块,每条序列至少要 1 个 KV 块 + %d 个状态块",
                    self.config.max_num_seqs, capacity, self.cache.num_blocks,
                    self.cache.num_state_groups)
            self.config.max_num_seqs = capacity
        if self.enable_log:
            self.cache.log_summary()

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        has_mm = any(seq.is_multimodal for seq in seqs)
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            if has_mm:
                positions.append(self.position_ids(seq, start, end))
            else:
                positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        if has_mm:
            positions = torch.cat(positions, dim=1).cuda(non_blocking=True)     # (3, 总 token 数)
        else:
            positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        linear_slot_ids = self.cache.prepare_linear_slot_ids(seqs)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables, linear_slot_ids)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        self.cache.check_linear_states(seqs)
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1 + (seq.inputs.rope_delta if seq.inputs is not None else 0))
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        linear_slot_ids = self.cache.prepare_linear_slot_ids(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens,
                    block_tables=block_tables, linear_slot_ids=linear_slot_ids)
        return input_ids, positions

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def position_ids(self, seq: Sequence, start: int, end: int) -> torch.Tensor:
        if seq.is_multimodal:
            return self.model.mm_position_ids(seq)[:, start:end] 
        return torch.arange(start, end).unsqueeze(0).expand(3, -1)

    def prepare_mm_inputs(self, seqs: list[Sequence]) -> dict:
        mm_seqs = [seq for seq in seqs if seq.is_multimodal]
        if not mm_seqs:
            return {}
        kwargs = {}
        for pixel_key, grid_key in (("pixel_values", "image_grid_thw"),
                                    ("pixel_values_videos", "video_grid_thw")):
            grids = [getattr(seq.inputs, grid_key) for seq in mm_seqs
                     if getattr(seq.inputs, grid_key) is not None]
            if not grids:
                continue
            pixels = [getattr(seq.inputs, pixel_key) for seq in mm_seqs
                      if getattr(seq.inputs, pixel_key) is not None]
            kwargs[grid_key] = torch.cat(grids, dim=0).cuda(non_blocking=True)
            kwargs[pixel_key] = self._scatter_mm(pixels, kwargs[grid_key])
        if "image_grid_thw" in kwargs:
            kwargs["image_visible_ranges"] = self._image_visible_ranges(
                mm_seqs, kwargs["image_grid_thw"])
        return kwargs

    def _image_visible_ranges(self, seqs: list[Sequence],
                              grid: torch.Tensor) -> list[tuple[int, int]] | None:
        """每张图在本步窗口里可见的特征行 (offset, count);全图都可见时返回 None。"""
        merge_unit = self.config.vision_config.spatial_merge_size ** 2
        ranges, lengths = [], []
        for seq in seqs:
            inputs = seq.inputs
            if inputs.image_grid_thw is None:
                continue
            rows = (inputs.image_grid_thw.prod(-1) // merge_unit).tolist()
            pads = inputs.image_pad_ranges
            assert pads is not None, (
                "多模态输入缺少 image_pad_ranges:只有 ModalInputs.build_inputs 会填它,"
                "手工构造 ModalInputs 时不能直接带 pixel_values")
            assert len(pads) == len(rows), (
                f"image_pad_ranges({pads}) 和 image_grid_thw 的 {len(rows)} 张图对不上")
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens
            for (pad_start, pad_len), row in zip(pads, rows):
                assert pad_len == row, (
                    f"占位符个数 {pad_len} 和 grid 推出的特征行数 {row} 对不上:"
                    "processor 出的 input_ids 和 grid 不是同一张图")
                begin = min(max(start - pad_start, 0), pad_len)
                finish = min(max(end - pad_start, 0), pad_len)
                ranges.append((begin, max(finish - begin, 0)))
                lengths.append(pad_len)
        assert len(ranges) == grid.shape[0], (
            f"算出 {len(ranges)} 张图,和 image_grid_thw 的 {grid.shape[0]} 行对不上")
        if all(count == length for (_, count), length in zip(ranges, lengths)):
            return None
        return ranges

    def _scatter_mm(self, pixels: list[torch.Tensor], grid: torch.Tensor) -> torch.Tensor:
        world = self.world_size
        rows = grid.prod(-1).tolist()
        counts = [shard.row_end - shard.row_start for shard in plan_shards(rows, world)]
        vision_config = self.config.vision_config
        patch_dim = (vision_config.in_channels * vision_config.temporal_patch_size
                     * vision_config.patch_size ** 2)
        dtype = self.model.mm_pixel_dtype

        if world == 1:
            assert pixels, "有 grid 却没有像素张量"
            return torch.cat(pixels, dim=0).to(device="cuda", dtype=dtype)

        device = torch.device("cuda", self.rank)
        if self.rank == 0:
            assert pixels, "rank0 上有 grid 却没有像素张量"
            # 按视觉塔的 dtype 传:塔里第一件事就是这个转换,提前做结果逐位相同、字节减半
            src = torch.cat(pixels, dim=0).to(device=device, dtype=dtype)
        else:
            src = torch.empty(0, patch_dim, dtype=dtype, device=device)
        mine = torch.empty(counts[self.rank], patch_dim, dtype=dtype, device=device)
        dist.all_to_all_single(
            mine, src,
            output_split_sizes=[counts[self.rank] if i == 0 else 0 for i in range(world)],
            input_split_sizes=counts if self.rank == 0 else [0] * world,
        )
        return mine