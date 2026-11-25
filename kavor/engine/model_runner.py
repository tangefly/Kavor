import pickle
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event

import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from kavor.config import Config
from kavor.engine.cache_pool import CachePool
from kavor.engine.mm_assembler import MMInputAssembler
from kavor.engine.mm_cache import MMCache
from kavor.engine.sequence import Sequence
from kavor.kernels.sampler import Sampler
from kavor.models import AutoModelForCausalLM
from kavor.utils.context import get_context, reset_context, set_context
from kavor.utils.logger import logger


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
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_config, use_tqdm=self.enable_log)
        self.sampler = Sampler()
        self.mm_cache = None
        self.allocate_kv_cache()
        self.mm_assembler = MMInputAssembler(
            self.model, config, self.mm_cache, self.rank, self.world_size, self.enable_log)
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
                f"TP worker payload is {n / 2**20:.1f} MB, over the "
                f"{len(self.shm.buf) / 2**20:.0f} MB shared buffer: keep large tensors "
                "out of args; multimodal pixels travel over NCCL "
                "(mm_inputs.scatter_pixels_to_ranks), not shared memory")
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
            # only rank 0 holds logits after the gather (see ParallelLMHead), and
            # sampling runs on rank 0 anyway
            if self.rank == 0:
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
            # as in warmup_model: non-zero ranks have no logits, sampling is rank-0 only
            if self.rank == 0:
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
                "Reserving %.2f GiB for activations: prefill %.2f GiB, "
                "decode %.2f GiB at bs=%d, CUDA graph pool %.2f GiB",
                reserve / 2**30, prefill_peak / 2**30,
                decode_peak / 2**30, max_bs, graph_pool / 2**30)
        return reserve

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
        mm_kwargs: dict | None = None,
        temperatures: torch.Tensor | None = None,
    ):
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
            if self.graph_captures_head:
                # lm_head + sampling live in the graph: temperatures go to a
                # persistent buffer and replay produces the tokens directly
                graph_vars["temperatures"][:bs].copy_(temperatures)
                graph.replay()
                return None
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        try:
            mm_kwargs = self.mm_assembler.prepare(seqs) if is_prefill else {}
            temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
            logits = self.run_model(input_ids, positions, is_prefill, mm_kwargs, temperatures)
        finally:
            self.mm_assembler.release_pins()
        self.cache.record_linear_states(seqs)
        if logits is None:    # the graph already includes lm_head + sampling
            bs = input_ids.size(0)
            token_ids = self.graph_vars["sampled_tokens"][:bs].tolist()
        else:
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
        temperatures = torch.ones(max_bs, dtype=torch.float32)
        sampled_tokens = torch.zeros(max_bs, dtype=torch.int64)
        # With TP>1 the dist.gather in ParallelLMHead can hang when captured, so
        # keep lm_head + sampling eager
        self.graph_captures_head = self.world_size == 1
        linear_slot_ids = (None if self.cache.linear_padding_slot is None else
                           torch.full((max_bs, self.cache.num_state_groups),
                                      self.cache.linear_padding_slot, dtype=torch.int64))
        self.graph_bs = [bs for bs in (1, 2, 4, 8) if bs <= max_bs]
        self.graph_bs += list(range(16, max_bs + 1, 16))
        if self.graph_bs[-1] != max_bs:
            self.graph_bs.append(max_bs)
        self.graphs = {}
        self.graph_pool = None

        def forward_with_head(bs):
            logits = self.model.compute_logits(outputs[:bs])
            self.sampler(logits, temperatures[:bs], out=sampled_tokens[:bs])

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
            if self.graph_captures_head:
                forward_with_head(bs)    # eager first: JIT and memory land in the pool
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
                if self.graph_captures_head:
                    forward_with_head(bs)
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
            temperatures=temperatures,
            sampled_tokens=sampled_tokens,
        )

    def allocate_kv_cache(self):
        """Build the page pool in two phases: probe the forward peak on a small
        temporary pool, then plan the real pool around the measured reserve.

        ``gpu_memory_utilization`` bounds weights + page pool + forward peak, and
        the pool size itself moves that peak (its largest term is the batch-sized
        linear-attention state), so plan optimistically, profile, then plan again.
        """
        # the mm pool must come first: the KV budget is derived from mem_get_info,
        # so a live mm pool is subtracted automatically
        self.mm_cache = MMCache.from_config(self.config)
        if self.mm_cache is not None and not hasattr(self.model, "get_mm_embeddings"):
            if self.enable_log:
                logger.warning(
                    "mm cache disabled: the model does not implement get_mm_embeddings")
            self.mm_cache = None
        if self.enable_log:
            if self.mm_cache is not None:
                self.mm_cache.log_summary()
            elif self.config.enable_mm_cache:
                reason = (
                    "the model has no vision tower" if self.config.vision_config is None
                    else "--mm-cache-size-gb is not positive")
                logger.warning("mm cache disabled: enable_mm_cache is on but %s", reason)

        self.cache = CachePool(self.model, self.config, self.world_size)

        # 1) optimistic plan without an activation reserve; real capacity never
        #    exceeds it, so it caps the batch size used for profiling
        self.cache.plan()
        probe_bs = min(self.config.max_num_seqs, 512, self.cache.max_concurrent())

        # 2) temporary pool just big enough for the prefill profiling pass, since
        #    prefill slot_mapping must land inside the pool
        seq_len, num_seqs = self.profile_prefill_shape()
        profile_blocks = (seq_len * num_seqs + self.cache.block_size - 1) // self.cache.block_size + 2
        self.cache.allocate(num_blocks=profile_blocks)

        # 3) measure the peak, then build the real pool (allocate frees the temp one)
        reserve = self.profile_activation_peak(probe_bs)
        layout = self.cache.allocate(reserve=reserve)

        self.block_size = layout.block_size
        Sequence.block_size = self.block_size
        self.config.kvcache_block_size = self.block_size
        self.config.num_kvcache_blocks = layout.num_blocks
        self.config.has_linear_attention = self.cache.has_linear_attention
        self.config.num_linear_state_groups = self.cache.num_state_groups

        # 4) clamp max_num_seqs to what the pool can really hold; the Scheduler and
        #    the CUDA graphs both use this value
        capacity = self.cache.max_concurrent()
        if capacity < self.config.max_num_seqs:
            if self.enable_log:
                logger.warning(
                    "Lowering max_num_seqs from %d to %d: the pool has %d blocks "
                    "and each sequence needs 1 KV block plus %d state blocks",
                    self.config.max_num_seqs, capacity, self.cache.num_blocks,
                    self.cache.num_state_groups)
            self.config.max_num_seqs = capacity
        if self.enable_log:
            self.cache.log_summary()

    def prepare_prefill(self, seqs: list[Sequence]):
        # sequences that hit a prefix snapshot or just crossed a chunk boundary:
        # copy the frozen state into their own mutable blocks first
        self.cache.copy_linear_states(seqs)
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
            positions = torch.cat(positions, dim=1).cuda(non_blocking=True)     # (3, total tokens)
        else:
            positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        linear_slot_ids = self.cache.prepare_linear_slot_ids(seqs)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables, linear_slot_ids)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        # sequences that just crossed a chunk boundary: copy the frozen snapshot
        # into the new block, then decode in place on it
        self.cache.copy_linear_states(seqs)
        self.cache.check_linear_states(seqs)
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1 + (seq.mm_inputs.rope_delta if seq.mm_inputs is not None else 0))
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
