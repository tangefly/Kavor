import asyncio
import logging
import queue
import threading
import traceback
from dataclasses import dataclass
from time import perf_counter

import torch

from kavor.engine.llm_engine import LLMEngine
from kavor.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


@dataclass
class RequestState:
    request_id: str
    queue: asyncio.Queue
    max_tokens: int
    arrival_time: float


class AsyncEngine:

    EXIT_TIMEOUT = 10.0

    def __init__(self, model: str, **kwargs):
        
        self.model_name = kwargs["model_name"] or model
        self.engine = LLMEngine(model, **kwargs)
        self.tokenizer = self.engine.tokenizer
        self.loop: asyncio.AbstractEventLoop | None = None

        self.incoming: queue.Queue = queue.Queue()
        self.new_request_event = threading.Event()
        self.tracker: dict[int, RequestState] = {}
        self._cursors: dict[int, int] = {}

        self._shutdown = False
        self._exiting = False
        self._exc: BaseException | None = None
        self._traceback: str | None = None

        self.thread = threading.Thread(target=self._engine_loop, name="kavor-engine", daemon=True)
        self.thread.start()

    async def add_request(self, request_id: str, prompt: str,
                          sampling_params: SamplingParams) -> asyncio.Queue:
        if self._exc is not None:
            raise RuntimeError(f"引擎线程已因异常退出: {self._exc!r}")
        if self._shutdown:
            raise RuntimeError("引擎已关闭,拒绝新请求")
        if self.loop is None:
            raise RuntimeError("AsyncEngine.loop 未注入,请在 lifespan 里设置")
        q: asyncio.Queue = asyncio.Queue()
        self.incoming.put((request_id, prompt, sampling_params, q))
        self.new_request_event.set()
        return q

    def is_alive(self) -> bool:
        return not self._shutdown and self._exc is None and self.thread.is_alive()

    def exit(self):
        if self._exiting:
            return
        self._exiting = True

        self._shutdown = True
        self.new_request_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=self.EXIT_TIMEOUT)

        if self.thread.is_alive():
            logger.warning("引擎线程 %.1fs 内未退出,跳过显式清理,交给 atexit", self.EXIT_TIMEOUT)
            return

        self.engine.exit()

        while True:
            try:
                _, _, _, q = self.incoming.get_nowait()
            except queue.Empty:
                break
            q.put_nowait(("finish", "abort"))
            q.put_nowait(None)

    def _push(self, aio_queue: asyncio.Queue, item):
        loop = self.loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(aio_queue.put_nowait, item)
        except RuntimeError:
            pass
    
    def _engine_loop(self):
        try:
            torch.cuda.set_device(self.engine.model_runner.rank)

            while not self._shutdown:
                self._drain_incoming()
                if self._shutdown:
                    break

                if self.engine.is_finished():
                    self.new_request_event.wait()
                    self.new_request_event.clear()
                    continue
                
                outputs, _ = self.engine.step()

                self._dispatch_running()
                self._dispatch_finished(outputs)
        except BaseException as e:
            self._exc = e
            self._traceback = traceback.format_exc()
            logger.exception("引擎线程异常退出")
        finally:
            self._fail_all_requests()

    def _drain_incoming(self):
        while True:
            try:
                request_id, prompt, sampling_params, q = self.incoming.get_nowait()
            except queue.Empty:
                break
            try:
                seq_id = self.engine.add_request(prompt, sampling_params)
            except BaseException:
                logger.exception("请求 %s 提交失败", request_id)
                self._push(q, ("finish", "error"))
                self._push(q, None)
                continue
            self.tracker[seq_id] = RequestState(
                request_id=request_id,
                queue=q,
                max_tokens=sampling_params.max_tokens,
                arrival_time=perf_counter(),
            )
            self._cursors[seq_id] = 0

    def _dispatch_running(self):
        for seq in self.engine.scheduler.running:
            seen = self._cursors.get(seq.seq_id, 0)
            num_new = seq.num_completion_tokens - seen
            if num_new <= 0:
                continue
            new = seq.token_ids[seq.num_prompt_tokens + seen:]
            self._cursors[seq.seq_id] = seen + num_new
            state = self.tracker.get(seq.seq_id)
            if state is not None:
                self._push(state.queue, ("delta", new))

    def _dispatch_finished(self, outputs: list[tuple[int, list[int]]]):
        for seq_id, token_ids in outputs:
            state = self.tracker.pop(seq_id, None)
            seen = self._cursors.pop(seq_id, 0)
            if state is None:
                continue
            new = token_ids[seen:]
            if new:
                self._push(state.queue, ("delta", new))
            reason = "length" if len(token_ids) >= state.max_tokens else "stop"
            self._push(state.queue, ("finish", reason))
            self._push(state.queue, None)

    def _fail_all_requests(self):
        for state in self.tracker.values():
            self._push(state.queue, ("finish", "abort"))
            self._push(state.queue, None)
        self.tracker.clear()
        self._cursors.clear()
