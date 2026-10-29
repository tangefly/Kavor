import atexit
import logging
import signal
from dataclasses import fields
from time import monotonic, perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoProcessor
import torch.multiprocessing as mp
import torch

from kavor.config import Config
from kavor.sampling_params import SamplingParams
from kavor.engine.sequence import Sequence
from kavor.engine.scheduler import Scheduler
from kavor.engine.model_runner import ModelRunner
from kavor.utils.logger import setup_logging
from .multimodal import ModelInputs

logger = logging.getLogger(__name__)

WORKER_EXIT_TIMEOUT = 10.0
WORKER_KILL_TIMEOUT = 5.0


def _worker_entry(config: Config, rank: int, event):
    setup_logging(rank=rank)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    ModelRunner(config, rank, event)


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        self._exiting = False
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=_worker_entry, args=(config, i, event))
            process.daemon = True
            process.start()
            self.ps.append(process)
            self.events.append(event)
        atexit.register(self.exit)
        try:
            self.model_runner = ModelRunner(config, 0, self.events)
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model, use_fast=True, trust_remote_code=config.trust_remote_code)
            self.processor = (
                AutoProcessor.from_pretrained(config.model, trust_remote_code=config.trust_remote_code)
                if getattr(config, "vision_config", None) is not None else None)
            config.eos = self.tokenizer.eos_token_id
            self.scheduler = Scheduler(config)
        except BaseException:
            self.exit()
            raise
        logger.info(
            "模型加载完成: %s | tensor_parallel_size=%d | max_model_len=%d | enforce_eager=%s",
            config.model, config.tensor_parallel_size, config.max_model_len, config.enforce_eager,
        )

    def exit(self):
        if self._exiting:
            return
        self._exiting = True
        workers_alive = all(p.is_alive() for p in self.ps)
        if getattr(self, "model_runner", None) is not None and workers_alive:
            self.model_runner.call("exit")
            del self.model_runner
            self._reap_workers(graceful=True)
        else:
            self._reap_workers(graceful=False)

    def _reap_workers(self, graceful: bool):
        if graceful:
            deadline = monotonic() + WORKER_EXIT_TIMEOUT
            for p in self.ps:
                p.join(timeout=max(0.0, deadline - monotonic()))
        for p in self.ps:
            if p.is_alive():
                logger.warning("TP 子进程 %s 未退出,发送 SIGTERM", p.name)
                p.terminate()
        deadline = monotonic() + WORKER_KILL_TIMEOUT
        for p in self.ps:
            p.join(timeout=max(0.0, deadline - monotonic()))
        for p in self.ps:
            if p.is_alive():
                logger.warning("TP 子进程 %s 对 SIGTERM 无响应,发送 SIGKILL", p.name)
                p.kill()
                p.join(timeout=2.0)

    def build_inputs(self, messages: list[dict], max_pixels: int | None = None,
                     processor_kwargs: dict | None = None,
                     **template_kwargs) -> ModelInputs:
        assert self.processor is not None, (
            f"{self.config.model} 没有 processor(AutoProcessor),不支持多模态输入")
        return ModelInputs.build_inputs(
            self.tokenizer, self.processor, messages, max_pixels=max_pixels,
            processor_kwargs=processor_kwargs, **template_kwargs)

    def add_request(
        self,
        inputs: ModelInputs,
        sampling_params: SamplingParams
    ) -> int:
        if len(inputs.token_ids) > self.config.max_num_batched_tokens:
            raise ValueError(
                f"prompt 有 {len(inputs.token_ids)} 个 token,超过 max_num_batched_tokens"
                f"({self.config.max_num_batched_tokens}):多模态序列的 prefill 不切分,排不下就会一直等。"
                "调大 --max-num-batched-tokens,或用更小的图/更少的帧")
        seq = Sequence(inputs, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self):
        scheduled_seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in scheduled_seqs) if is_prefill else -len(scheduled_seqs)
        token_ids = self.model_runner.call("run", scheduled_seqs, is_prefill)
        if is_prefill:
            for seq in scheduled_seqs:
                seq.first_token_time = perf_counter()
        self.scheduler.postprocess(scheduled_seqs, token_ids, is_prefill)
        for seq in scheduled_seqs:
            if seq.is_finished:
                seq.finish_time = perf_counter()
        outputs = [(seq.seq_id, seq.completion_token_ids, seq.first_token_time, seq.finish_time) for seq in scheduled_seqs if seq.is_finished]
        
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()
    
    def normalize_inputs(
        self, 
        inputs: list[str] | list[list[int]] | list[ModelInputs]
    ) -> list[ModelInputs]:
        normalized = []
        for item in inputs:
            if isinstance(item, ModelInputs):
                normalized.append(item)
            elif isinstance(item, str):
                token_ids = self.tokenizer.encode(item)
                normalized.append(ModelInputs(token_ids, item))
            elif isinstance(item, int):
                normalized.append(ModelInputs(torch.tensor(item), None))
            else:
                raise RuntimeError("Not supported data type!")
        return normalized

    def generate(
        self,
        inputs: list[str] | list[list[int]] | list[ModelInputs],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        
        inputs = self.normalize_inputs(inputs)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(inputs)
        for item, sp in zip(inputs, sampling_params):
            self.add_request(item, sp)
        
        outputs = {}
        
        prefill_throughput = 0.
        decode_throughput = 0.
        
        pbar = tqdm(total=len(inputs), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids, _, _ in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
