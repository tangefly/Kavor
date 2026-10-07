import atexit
import logging
import signal
from dataclasses import fields
from time import monotonic, perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from kavor.config import Config
from kavor.sampling_params import SamplingParams
from kavor.engine.sequence import Sequence
from kavor.engine.scheduler import Scheduler
from kavor.engine.model_runner import ModelRunner

logger = logging.getLogger(__name__)

WORKER_EXIT_TIMEOUT = 10.0   # 等 TP 子进程优雅退出的上限(秒)
WORKER_KILL_TIMEOUT = 5.0    # SIGTERM 之后再等这么久,然后 SIGKILL


def _worker_entry(config: Config, rank: int, event):
    """TP 子进程入口(rank > 0)。

    Ctrl-C 不是只发给父进程 —— 终端会把 SIGINT 发给**整个前台进程组**,TP 子进程
    也在里面。如果子进程跟着一起死,父进程 ModelRunner.exit() 里的 dist.barrier()
    就永远等不到对端:进程挂死、显存不释放、只能 kill -9。

    所以子进程忽略 SIGINT,退出统一由父进程编排(父进程发退出信号 → 子进程跳出
    loop() 走完 barrier → 父进程 join)。
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    ModelRunner(config, rank, event)


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        self._exiting = False
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=_worker_entry, args=(config, i, event))
            process.daemon = True   # 父进程意外退出时由解释器兜底终止,不留孤儿进程
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # 必须注册在最慢的模型加载之前:加载途中被 Ctrl-C 打断时,
        # 上面已经起好的 TP 子进程需要有人收尸
        atexit.register(self.exit)
        try:
            self.model_runner = ModelRunner(config, 0, self.events)
            self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
            config.eos = self.tokenizer.eos_token_id
            self.scheduler = Scheduler(config)
        except BaseException:
            # 加载失败 / 被 Ctrl-C 打断:别把 TP 子进程留成孤儿
            self.exit()
            raise

    def exit(self):
        """停引擎并回收所有 TP 子进程。幂等,任何线程、任何阶段调用都安全。"""
        if self._exiting:
            return
        self._exiting = True
        # 只要有一个子进程已经死了,就不能走优雅路径:ModelRunner.exit() 里的
        # dist.barrier() 会永远等不到对端,把整个进程挂死在 teardown 里
        workers_alive = all(p.is_alive() for p in self.ps)
        if getattr(self, "model_runner", None) is not None and workers_alive:
            self.model_runner.call("exit")   # 通知各 rank 跳出 loop()
            del self.model_runner
            self._reap_workers(graceful=True)
        else:
            # 要么 rank 0 自己都没构造完(子进程在等一个永远不来的信号),
            # 要么已经有人先死了 —— 都只能强杀
            self._reap_workers(graceful=False)

    def _reap_workers(self, graceful: bool):
        """回收 TP 子进程:等 → SIGTERM → SIGKILL,三级兜底,不挂死也不留孤儿。"""
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

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> int:
        """提交请求,返回 seq_id(服务化需要它来关联输出队列)。"""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
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
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
