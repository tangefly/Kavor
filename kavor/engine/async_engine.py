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
    """一个在途请求的状态。只允许引擎线程读写。"""

    request_id: str
    queue: asyncio.Queue
    max_tokens: int          # 判 finish_reason 用:结束分支拿不到 Sequence 对象
    arrival_time: float


class AsyncEngine:

    EXIT_TIMEOUT = 10.0  # 等引擎线程退出的上限(秒)

    def __init__(self, model: str, **kwargs):
        
        self.engine = LLMEngine(model, **kwargs)
        self.tokenizer = self.engine.tokenizer
        self.loop: asyncio.AbstractEventLoop | None = None  # 由 lifespan 注入

        self.incoming: queue.Queue = queue.Queue()
        self.new_request_event = threading.Event()
        self.tracker: dict[int, RequestState] = {}  # seq_id -> 请求状态(引擎线程独占)
        self._cursors: dict[int, int] = {}          # seq_id -> 已推送的 completion token 数

        self._shutdown = False
        self._exiting = False
        self._exc: BaseException | None = None
        self._traceback: str | None = None

        self.thread = threading.Thread(target=self._engine_loop, name="kavor-engine", daemon=True)
        self.thread.start()

    async def add_request(self, request_id: str, prompt: str,
                          sampling_params: SamplingParams) -> asyncio.Queue:
        """提交请求,立刻返回该请求的输出队列(不阻塞等结果)。"""
        if self._exc is not None:
            raise RuntimeError(f"引擎线程已因异常退出: {self._exc!r}")
        if self._shutdown:
            raise RuntimeError("引擎已关闭,拒绝新请求")
        if self.loop is None:
            # 没走 lifespan 就构造了引擎。直接报错,别静默退化成"token 推不出来"
            raise RuntimeError("AsyncEngine.loop 未注入,请在 lifespan 里设置")
        q: asyncio.Queue = asyncio.Queue()
        self.incoming.put((request_id, prompt, sampling_params, q))
        self.new_request_event.set()  # 唤醒可能正睡在 wait() 上的引擎线程
        return q

    def is_alive(self) -> bool:
        """引擎线程是否健康(/health 探针用)。"""
        return not self._shutdown and self._exc is None and self.thread.is_alive()

    def exit(self):
        """停引擎线程并收尾。幂等。由 lifespan 在事件循环线程里调用。"""
        if self._exiting:
            return
        self._exiting = True

        self._shutdown = True
        self.new_request_event.set()  # 唤醒可能正睡在 wait() 上的引擎线程
        if self.thread.is_alive():
            self.thread.join(timeout=self.EXIT_TIMEOUT)

        if self.thread.is_alive():
            # 引擎线程卡住了(大概率是在 step() 里等 GPU)。此时不能拆 ModelRunner:
            # 会和 step() 抢 dist/shm,报一堆 NCCL 错。交给 LLMEngine 里的 atexit 兜底。
            logger.warning("引擎线程 %.1fs 内未退出,跳过显式清理,交给 atexit", self.EXIT_TIMEOUT)
            return

        # 线程已死,不再有并发调用,从任何线程拆引擎都安全。
        # 在途请求的哨兵由 _engine_loop 的 finally 负责推(在 join 返回前已完成)。
        self.engine.exit()

        # 极窄的关闭窗口:请求在引擎线程跑完 finally 之后才入队。
        # 此时我们就在事件循环线程上,直接 put_nowait 即可(不用绕 _push)。
        while True:
            try:
                _, _, _, q = self.incoming.get_nowait()
            except queue.Empty:
                break
            q.put_nowait(("finish", "abort"))
            q.put_nowait(None)

    def _push(self, aio_queue: asyncio.Queue, item):
        """跨线程往 asyncio.Queue 塞数据的唯一通道。

        为什么不能直接 aio_queue.put_nowait:asyncio 对象不是线程安全的;而且就算数据
        塞进去了,正睡在 epoll_wait 里的事件循环也不会被唤醒 —— 表现为"token 明明已经
        产出,SSE 却半天不吐字"。call_soon_threadsafe 会把回调排进就绪队列并唤醒循环。
        """
        loop = self.loop
        if loop is None or loop.is_closed():
            return  # 关闭期:此刻已经没有消费者了,静默丢弃好过让引擎线程收尾时抛异常
        try:
            loop.call_soon_threadsafe(aio_queue.put_nowait, item)
        except RuntimeError:
            pass  # is_closed() 检查和真正调用之间仍有窗口

    def _engine_loop(self):
        try:
            # CUDA 的 current device 是线程局部的:ModelRunner 构造时设的 device 只对
            # 主线程生效。rank 0 恰好是 device 0,这里显式设一次,不依赖"新线程默认 0"。
            torch.cuda.set_device(self.engine.model_runner.rank)

            while not self._shutdown:
                # ① 收割入口队列(必须 drain 到空:Event 只负责叫醒,不传递数据)
                self._drain_incoming()
                if self._shutdown:
                    break

                # ② 没活儿干就挂起,绝不空转轮询。
                #    顺序必须是 wait() 再 clear():反过来会丢唤醒,线程永久卡死。
                if self.engine.is_finished():
                    self.new_request_event.wait()
                    self.new_request_event.clear()
                    continue

                # ③ 跑一步(阻塞的 CUDA 调用,只在这个线程里发生)。
                #    必须先判 is_finished():scheduler.schedule() 末尾有
                #    assert scheduled_seqs,空调度会直接崩。
                outputs, _ = self.engine.step()

                # ④ 派发增量。先 running 再 finished:结束的 seq 已不在 running 里,
                #    两者天然互斥,不会重复推送同一段 token。
                self._dispatch_running()
                self._dispatch_finished(outputs)
        except BaseException as e:
            # 线程里的异常不会传播到主线程,不兜住就是"静默死亡 + /health 还报 200"
            self._exc = e
            self._traceback = traceback.format_exc()
            logger.exception("引擎线程异常退出")
        finally:
            self._fail_all_requests()

    def _drain_incoming(self):
        """把入口队列里的新请求转交给 LLMEngine。只允许引擎线程调用。"""
        while True:
            try:
                request_id, prompt, sampling_params, q = self.incoming.get_nowait()
            except queue.Empty:
                break
            try:
                seq_id = self.engine.add_request(prompt, sampling_params)
            except BaseException:
                # 单个请求的错(比如 tokenize 失败)不该弄死整个引擎
                logger.exception("请求 %s 提交失败", request_id)
                self._push(q, ("finish", "error"))
                self._push(q, None)
                continue
            # add_request 返回后立刻登记:同一线程内不会插入 step(),
            # 所以不存在"token 已产出但还没登记"的窗口
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
