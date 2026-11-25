import atexit
import logging
import signal
from time import monotonic, perf_counter

import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoProcessor, AutoTokenizer

from kavor.config import Config
from kavor.engine.model_runner import ModelRunner
from kavor.engine.scheduler import Scheduler
from kavor.engine.sequence import Sequence
from kavor.sampling_params import SamplingParams
from kavor.utils.logger import setup_logging

from .multimodal import PromptInputs

logger = logging.getLogger(__name__)

WORKER_EXIT_TIMEOUT = 10.0
WORKER_KILL_TIMEOUT = 5.0


def _worker_entry(config: Config, rank: int, event):
    setup_logging(rank=rank)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    ModelRunner(config, rank, event)


class LLMEngine:

    def __init__(self, model, **kwargs):
        self.config = Config(model, **kwargs)
        Sequence.block_size = self.config.kvcache_block_size
        self.worker_processes = []
        self.events = []
        self._exiting = False
        self.step_stats: list[dict] = []
        self.request_stats: list[dict] = []
        ctx = mp.get_context("spawn")
        
        for i in range(1, self.config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=_worker_entry, args=(self.config, i, event))
            process.daemon = True
            process.start()
            self.worker_processes.append(process)
            self.events.append(event)
        atexit.register(self.exit)
        
        try:
            self.model_runner = ModelRunner(self.config, 0, self.events)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model, use_fast=True,
                trust_remote_code=self.config.trust_remote_code
            )
            self.processor = (
                AutoProcessor.from_pretrained(
                    self.config.model, 
                    trust_remote_code=self.config.trust_remote_code
                )
                if getattr(self.config, "vision_config", None) is not None else None
            )
            self.config.eos = self.tokenizer.eos_token_id
            self.scheduler = Scheduler(self.config)
        except BaseException:
            self.exit()
            raise
        logger.info("Engine initialized with config: %s", self.config)

    def exit(self):
        if self._exiting:
            return
        self._exiting = True
        workers_alive = all(
            process.is_alive() for process in self.worker_processes)
        if getattr(self, "model_runner", None) is not None and workers_alive:
            self.model_runner.call("exit")
            del self.model_runner
            self._shutdown_tp_workers(graceful=True)
        else:
            self._shutdown_tp_workers(graceful=False)

    def _shutdown_tp_workers(self, graceful: bool):
        """Stop the TP worker processes, escalating from join to SIGTERM to SIGKILL.

        With ``graceful=True`` the workers have already been told to exit (via
        ``model_runner.call("exit")``), so give them WORKER_EXIT_TIMEOUT to wind
        down before signalling anything.
        """
        if graceful:
            deadline = monotonic() + WORKER_EXIT_TIMEOUT
            for process in self.worker_processes:
                process.join(timeout=max(0.0, deadline - monotonic()))
        for process in self.worker_processes:
            if process.is_alive():
                logger.warning("TP worker %s did not exit, sending SIGTERM", process.name)
                process.terminate()
        deadline = monotonic() + WORKER_KILL_TIMEOUT
        for process in self.worker_processes:
            process.join(timeout=max(0.0, deadline - monotonic()))
        for process in self.worker_processes:
            if process.is_alive():
                logger.warning("TP worker %s did not respond to SIGTERM, sending SIGKILL", process.name)
                process.kill()
                process.join(timeout=2.0)

    def build_inputs(
        self, 
        messages: list[dict], 
        max_pixels: int | None = None,
        processor_kwargs: dict | None = None,
        **template_kwargs
    ) -> PromptInputs:
        assert self.processor is not None, (
            f"{self.config.model} has no processor (AutoProcessor); "
            "multimodal inputs are not supported")
        return PromptInputs.build_inputs(
            self.tokenizer, 
            self.processor, 
            messages, 
            max_pixels=max_pixels,
            processor_kwargs=processor_kwargs, 
            **template_kwargs
        )

    def add_request(
        self,
        inputs: PromptInputs,
        sampling_params: SamplingParams,
        request_id: str | None = None,
    ) -> int:
        if len(inputs.token_ids) > self.config.max_num_batched_tokens:
            raise ValueError(
                f"prompt has {len(inputs.token_ids)} tokens, more than "
                f"max_num_batched_tokens ({self.config.max_num_batched_tokens}): "
                "multimodal prefill is not chunked, so an oversized prompt would "
                "wait forever. Raise --max-num-batched-tokens, or use smaller "
                "images / fewer frames")
        seq = Sequence(inputs, sampling_params)
        seq.request_id = request_id
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self):
        scheduled_seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", scheduled_seqs, is_prefill)
        if is_prefill:
            for seq in scheduled_seqs:
                seq.stats.first_token_time = perf_counter()
        self.scheduler.postprocess(scheduled_seqs, token_ids, is_prefill)
        for seq in scheduled_seqs:
            if seq.is_finished:
                self.log_request_stats(seq, perf_counter())
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduled_seqs if seq.is_finished]

        return outputs

    def is_finished(self):
        return self.scheduler.is_finished()
    
    def normalize_inputs(
        self, 
        inputs: list[str] | list[list[int]] | list[PromptInputs]
    ) -> list[PromptInputs]:
        normalized = []
        for item in inputs:
            if isinstance(item, PromptInputs):
                normalized.append(item)
            elif isinstance(item, str):
                token_ids = self.tokenizer.encode(item)
                normalized.append(PromptInputs(token_ids, item))
            elif isinstance(item, list):
                normalized.append(PromptInputs(item, None))
            elif isinstance(item, int):
                normalized.append(PromptInputs(torch.tensor(item), None))
            else:
                raise RuntimeError("Not supported data type!")
        return normalized

    def generate(
        self,
        inputs: list[str] | list[list[int]] | list[PromptInputs],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        # In profile mode keep only the latest generate()'s stats (a warmup call clears them)
        if self.config.profile:
            self.step_stats.clear()
            self.request_stats.clear()
        inputs = self.normalize_inputs(inputs)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(inputs)
        for item, sp in zip(inputs, sampling_params):
            self.add_request(item, sp)
        
        outputs = {}

        pbar = tqdm(total=len(inputs), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        step = self.profile_step if self.config.profile else self.step
        while not self.is_finished():
            for seq_id, token_ids in step():
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs

    def profile_step(self):
        """Step variant that also records timings (used when config.profile=True).

        Per-step timings go to self.step_stats, per-request timings of finished
        requests go to self.request_stats.
        """
        start = perf_counter()
        scheduled_seqs, is_prefill = self.scheduler.schedule()
        # postprocess zeroes num_scheduled_tokens, so read the count first
        num_tokens = sum(seq.num_scheduled_tokens for seq in scheduled_seqs)
        forward = perf_counter()
        token_ids = self.model_runner.call("run", scheduled_seqs, is_prefill)
        forward_time = perf_counter() - forward
        if is_prefill:
            for seq in scheduled_seqs:
                seq.stats.first_token_time = perf_counter()
        self.scheduler.postprocess(scheduled_seqs, token_ids, is_prefill)
        now = perf_counter()
        self.step_stats.append({
            "ts": now,
            "prefill": is_prefill,
            "tokens": num_tokens,
            "step_time": now - start,
            "forward_time": forward_time,
        })
        for seq in scheduled_seqs:
            if seq.is_finished:
                stats = seq.stats
                stats.finish_time = now
                self.request_stats.append({
                    "seq_id": seq.seq_id,
                    "request_id": seq.request_id,
                    "num_prompt_tokens": seq.num_prompt_tokens,
                    "num_completion_tokens": seq.num_completion_tokens,
                    "arrival_time": stats.arrival_time,
                    "prefill_start_time": stats.prefill_start_time,
                    "first_token_time": stats.first_token_time,
                    "finish_time": stats.finish_time,
                    "prefix_cache_hit_tokens": stats.prefix_cache_hit_tokens,
                })
                self.log_request_stats(seq, now)
        outputs = [(seq.seq_id, seq.completion_token_ids)
                   for seq in scheduled_seqs if seq.is_finished]

        return outputs

    def log_request_stats(self, seq: Sequence, finish: float):
        stats = seq.stats
        prompt = seq.num_prompt_tokens
        completion = seq.num_completion_tokens
        first = stats.first_token_time

        label = f"seq {seq.seq_id}"
        if seq.request_id:
            label += f" ({seq.request_id})"
        parts = [f"{label} finished", f"prompt length: {prompt} tokens"]
        prefix_hit = stats.prefix_cache_hit_tokens
        if prefix_hit is not None:
            parts.append(
                f"prefix cache hit rate: {prefix_hit}/{prompt}"
                f" ({100.0 * prefix_hit / prompt:.1f}%)"
            )
        parts.append(f"output length: {completion} tokens")
        if stats.mm_total_items:
            parts.append(
                f"mm cache hit items: {stats.mm_hit_items}/{stats.mm_total_items}")
            parts.append(
                f"mm cache hit rate:"
                f" {100.0 * stats.mm_hit_items / stats.mm_total_items:.1f}%"
            )

        decode_tokens = completion - 1
        decode_time = finish - first if first is not None else 0.0
        if first is not None:
            parts.append(f"TTFT: {first - stats.arrival_time:.2f}s")
        if decode_tokens > 0 and decode_time > 0:
            parts.append(f"TPOT: {1000.0 * decode_time / decode_tokens:.1f}ms")
        if first is not None and stats.prefill_start_time is not None:
            prefill = first - stats.prefill_start_time
            if prefill > 0:
                computed = prompt - (prefix_hit or 0)
                parts.append(f"prefill throughput: {computed / prefill:.0f} tok/s")
        if decode_tokens > 0 and decode_time > 0:
            parts.append(f"decode throughput: {decode_tokens / decode_time:.0f} tok/s")
        parts.append(f"elapsed: {finish - stats.arrival_time:.2f}s")

        logger.info("%s", ", ".join(parts))