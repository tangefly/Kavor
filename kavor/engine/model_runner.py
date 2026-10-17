import pickle

import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from kavor.config import Config
from kavor.engine.cache_pool import CachePool
from kavor.engine.multimodal import MultiModalInputs
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
        # self.warmup_model()
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
                "(多模态的像素张量应该在 MultiModalInputs.for_workers 里就被丢掉,"
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

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence(MultiModalInputs([0] * seq_len, "")) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool, mm_kwargs: dict | None = None):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
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
        max_bs = min(self.config.max_num_seqs, 512)
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
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
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
        """建页池,并把布局参数发布给调度器(rank 0 的 Scheduler 依赖这些值)。"""
        self.cache = CachePool(self.model, self.config, self.world_size)
        layout = self.cache.allocate()
        self.block_size = layout.block_size
        Sequence.block_size = self.block_size
        self.config.kvcache_block_size = self.block_size
        self.config.num_kvcache_blocks = layout.num_blocks
        self.config.has_linear_attention = self.cache.has_linear_attention
        self.config.num_linear_state_groups = self.cache.num_state_groups
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
        has_mm = any(seq.mm is not None for seq in seqs)
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
            positions.append(len(seq) - 1 + (seq.mm.rope_delta if seq.mm is not None else 0))
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
        if seq.mm is None:
            return torch.arange(start, end).unsqueeze(0).expand(3, -1)
        return self.model.mm_position_ids(seq)[:, start:end]

    def prepare_mm_inputs(self, seqs: list[Sequence]) -> dict:
        mm_seqs = [seq for seq in seqs if seq.mm is not None]
        if not mm_seqs:
            return {}
        kwargs = {}
        for pixel_key, grid_key in (("pixel_values", "image_grid_thw"),
                                    ("pixel_values_videos", "video_grid_thw")):
            grids = [getattr(seq.mm, grid_key) for seq in mm_seqs
                     if getattr(seq.mm, grid_key) is not None]
            if not grids:
                continue
            pixels = [getattr(seq.mm, pixel_key) for seq in mm_seqs
                      if getattr(seq.mm, pixel_key) is not None]
            kwargs[grid_key] = torch.cat(grids, dim=0).cuda(non_blocking=True)
            kwargs[pixel_key] = self._scatter_mm(pixels, kwargs[grid_key])
        return kwargs

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