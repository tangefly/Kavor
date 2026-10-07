# AsyncEngine 设计讲解（写给异步新手）

本文讲清楚 `kavor/engine/async_engine.py` 该怎么设计，以及每个设计决策背后的原因。对应 [`fastapi_serving.md`](fastapi_serving.md) 第 2 步（增量输出）和第 3 步（AsyncEngine 封装层），并把那份文档里没讲透的地方补齐。

**阅读姿势**：§1-2 是心智模型，先看；§3-8 是实现设计，照着写；§9-11 是坑；§12 是验证清单。

---

## 1. 先建立心智模型：为什么需要一个"异步引擎"

### 1.1 asyncio 不是多线程

asyncio 的核心是**一个线程 + 一个事件循环**。事件循环不停地问 "谁现在可以往下走了？"（哪个 socket 可读了、哪个 `sleep` 到点了、哪个队列有数据了），然后挨个推进。

所以 `async def` 里的代码**默认还是同步执行的**，`await` 才是唯一的"让出"点：

```python
async def handler():
    a = 1 + 1          # 独占地跑
    await q.get()      # ← 让出：这期间事件循环去跑别的 handler
    b = 2 + 2          # 又独占地跑
```

**推论（你以后 90% 的 bug 都出在这里）**：只要一个协程在两次 `await` 之间做了耗时的**阻塞**操作，整个事件循环里所有协程都会卡住 —— 不是这一个请求慢，是**全部请求**都慢。

而 `LLMEngine.step()` 恰恰就是这种操作：它调 `model_runner.call("run", ...)`，本质是一次 CUDA kernel 同步执行 + 等待，几十毫秒起步，期间**没有任何可以 `await` 的点**。

### 1.2 那能不能把 `step()` 改写成 `async def`？

不能。`async def` 不会让阻塞代码变快或变并行，它只是"允许里面有 `await`"。你没法 `await` 一个 CUDA kernel —— GPU 的同步执行是个死等的 C 调用。给 `step()` 套上 `async` 然后直接在 handler 里 `await engine.step()`，效果和同步调用**完全一样**，事件循环照样冻住。

### 1.3 真正的解法：把阻塞的东西挪出事件循环

现代 LLM 服务框架（vLLM 的 `AsyncLLMEngine`、TGI 等）都是同一个套路：

> **不要试图"异步化"引擎。让引擎跑在它该待的地方（自己的线程），让 HTTP 跑在事件循环线程，两者之间用管道通信。**

`AsyncEngine` 就是一个**适配器（adapter）**：它对 async 世界暴露 `async def add_request(...) -> asyncio.Queue` 这样友好的接口，内部则把活儿丢给自己的后台线程，再把结果搬运回来。

所以真正要学的不是 "asyncio 语法"，而是这三件事：

1. **边界**：哪段代码在哪个线程跑（不可越界）
2. **管道**：两个世界之间怎么安全传数据
3. **时序**：空闲时怎么等、结束时怎么收尸

---

## 2. 三个世界 + 两条管道

```
  ┌──────────────────────────────┐
  │ 主线程：asyncio 事件循环       │   FastAPI handler / SSE 生成器
  │  - await engine.add_request() │   消费 asyncio.Queue
  └───────┬──────────────▲───────┘
          │              │
     incoming 队列    asyncio.Queue
     (queue.Queue)   + loop.call_soon_threadsafe
          │              │
  ┌───────▼──────────────┴───────┐
  │ 引擎线程：AsyncEngine 主循环   │   ★ 新增
  │  - engine.add_request()      │   ← 只有这个线程能碰 Scheduler
  │  - engine.step()             │
  └───────┬──────────────────────┘
          │  已有的 SharedMemory + Event 机制
  ┌───────▼──────────────────────┐
  │ TP 子进程：ModelRunner(rank>0) │   已有，不用动
  └──────────────────────────────┘
```

### 2.1 线程边界（最重要的一条规则）

**`Scheduler` 里没有任何锁**（`kavor/engine/scheduler.py` 全是裸的 `deque` 操作、`block_manager` 的 `ref_count` 加减）。所以：

| 只能在**引擎线程**调用 | 可以在**任意线程**调用 |
|---|---|
| `engine.add_request()` | 读 `engine.tokenizer`（纯 Python，无状态） |
| `engine.step()` | `engine.tokenizer.apply_chat_template(...)` |
| `engine.scheduler.*` 的任何读写 | 引擎内部状态（除 `is_finished()` 外）都别碰 |
| `engine.exit()`（join TP 子进程） | |

HTTP handler 想提交请求？**只能通过 `incoming` 队列**，由引擎线程代为调用。这不是洁癖 —— 两个线程同时进 `Scheduler.schedule()` 会直接把 `block_manager` 的 `ref_count` 算错，KV cache block 泄漏或重复分配，跑几轮就 OOM 或输出乱码，而且是概率性的、极难复现。

### 2.2 为什么用线程，不用进程？

理论上引擎可以像 TP 子进程一样单独跑一个进程（vLLM V1 的 `EngineCore` 就是这么干的）。但这一版用线程更划算：

- Python 的 GIL 在**进入 CUDA 调用时会释放**。`step()` 的时间 99% 花在 GPU kernel 里，那段时间引擎线程不持 GIL，事件循环线程照跑不误。
- 线程共享内存，`Sequence`、`tokenizer` 直接引用传递，不用序列化。
- 换成进程要重写一套 IPC 协议（vLLM V1 为此写了 `EngineCoreProc` + ZMQ），对第一版是过度设计。

**注意一个非 CUDA 的例外**：`engine.add_request()` 里的 `tokenizer.encode()`、`scheduler.schedule()` 是纯 Python，跑的时候是持 GIL 的。请求量大、prompt 很长时，会看到 HTTP 侧有轻微抖动。第一版忽略它，知道有这回事就行。

### 2.3 两条管道为什么不一样

| 方向 | 用什么 | 为什么 |
|---|---|---|
| 事件循环 → 引擎线程 | 普通 `queue.Queue`（`incoming`） | `queue.Queue` 天生线程安全，两边随便调。**不需要唤醒机制**，因为引擎线程是主动去 drain 它的 |
| 引擎线程 → 事件循环 | `asyncio.Queue` + `loop.call_soon_threadsafe` | asyncio 的对象**不是**线程安全的，必须绕道（见 §5） |

---

## 3. 契约：要实现的接口

契约的**消费侧**在 `api_server.py`：`consume_all()`（非流式）和 `stream_sse()`（SSE，`kavor/entrypoints/openai/api_server.py:40`）就是要对齐的消费者：

| 成员 | 语义 |
|---|---|
| `__init__(model, **kwargs)` | kwargs 是 Config 字段（`tensor_parallel_size` / `max_model_len` / `max_num_seqs` / `gpu_memory_utilization` / `enforce_eager`），直接透传给 `LLMEngine` |
| `await add_request(request_id, prompt, sampling_params) -> asyncio.Queue` | 提交请求，立刻返回该请求的输出队列（**不阻塞等结果**） |
| 队列消息协议 | `("delta", list[int])` → 可多次；`("finish", str)` → 一次；`None` → 哨兵，消费端据此退出 |
| `.tokenizer` | 暴露 `LLMEngine.tokenizer`，服务层拿来 decode / chat template |
| `.loop` | 事件循环引用，lifespan 里注入（`engine.loop = asyncio.get_running_loop()`） |
| `.is_alive() -> bool` | 引擎线程还活着吗（`/health` 探针用） |
| `.exit()` | 停线程 + 收尾（内部调 `LLMEngine.exit()` join TP 子进程） |

`is_alive()` 必须能反映"引擎线程已经死了"（比如 `step()` 里抛异常挂了），不能敷衍地永远返回 `True` —— 否则 `/health` 探针永远绿着，K8s 不会重启你的 Pod，而实际上所有请求都在挂死。

---

## 4. 数据结构设计

```python
@dataclass
class RequestState:
    request_id: str
    queue: asyncio.Queue      # 推给 HTTP 侧的输出队列
    max_tokens: int           # ★ 必须存：判 finish_reason 用（见 §7.1）
    arrival_time: float
```

⚠️ **别想着在这里放 `num_prompt_tokens`** —— `LLMEngine.add_request()` 内部才 tokenize（`llm_engine.py:44`），它既不返回 token 数、也不返回 `Sequence`；引擎线程手里只有原始字符串。usage 统计要这个数的话，只有两条路：让 `add_request` 多返回一个值，或者把手里这个 `prompt` 再 `encode` 一遍（贵）。第一版先不做 usage 就别存，`prompt` 全文也别存（长 prompt 常驻内存不划算）。

`AsyncEngine` 的字段：

| 字段 | 类型 | 作用 |
|---|---|---|
| `engine` | `LLMEngine` | 被包装的同步引擎 |
| `tokenizer` | — | 直接指向 `engine.tokenizer` |
| `loop` | `AbstractEventLoop \| None` | 主事件循环，lifespan 注入 |
| `incoming` | `queue.Queue` | 事件循环 → 引擎线程的请求入口 |
| `new_request_event` | `threading.Event` | 空闲时挂起引擎线程，新请求来了唤醒 |
| `tracker` | `dict[int, RequestState]` | **seq_id** → 请求状态（注意 key 是 seq_id，不是 request_id） |
| `thread` | `threading.Thread` | daemon=True |
| `_exc` / `_shutdown` | — | 异常记录 / 关闭标志 |

**为什么 `tracker` 用 `seq_id` 做 key**：引擎线程每步拿到的只有 `seq_id`（`step()` 的返回值、`scheduler.running` 里的 `Sequence`），拿它查表是热路径，必须 O(1)。反向的 `request_id → seq_id` 映射只在 abort 时用一次，可以再存一个 `dict[str, int]`，或者线性扫 `tracker`（请求数量级几百，无所谓）。

**为什么游标 `cursors: dict[int, int]` 是主循环的局部变量**而不是实例字段：它只被引擎线程读写，没有跨线程可见性需求，放局部变量里反而更安全（防止将来有人从别的线程误读）。

---

## 5. 跨线程推数据：为什么必须 `loop.call_soon_threadsafe`

这是整个设计里最容易写错的一处。

### 5.1 直接 `asyncio_queue.put_nowait(item)` 会怎样

看起来没事：`asyncio.Queue` 内部就是个 `collections.deque`，`append` 是原子的。数据确实进得去。

**但消费端不会醒。** 事件循环此刻正阻塞在 `epoll_wait()`（等 IO 事件）里睡觉。`await q.get()` 的协程是被 `Queue.get()` 内部的一个 `Future` 挂起的，`put_nowait` 会 `set_result` 唤醒这个 Future —— 然而"唤醒 Future"只是把它塞进事件循环的就绪队列，**没有任何东西会去叫醒正在 epoll_wait 的事件循环**。于是：

- 主循环可能在跑别的逻辑，或者
- 主循环睡在 `epoll_wait` 里直到下一个 socket 事件才醒

结果就是**你的 token 已经躺在队列里了，但 SSE 半天不吐字**，直到恰好有别的网络事件把循环戳醒。典型现象：单请求压测偶尔正常，并发或空闲时首 token 延迟莫名其妙几秒到几十秒。

另外，`asyncio.Queue` 内部有 `_unfinished_tasks`、`_finished` 之类状态，asyncio 的文档明确说它**不是线程安全的**，跨线程用属于未定义行为。

### 5.2 `call_soon_threadsafe` 干了两件事

```python
self.loop.call_soon_threadsafe(aio_queue.put_nowait, item)
```

1. 把 `aio_queue.put_nowait(item)` 这个**回调排进事件循环的就绪队列**
2. **通过 self-pipe/socketpair 唤醒阻塞中的事件循环**

于是 `put_nowait` 永远在事件循环线程里执行，线程安全问题消失；同时循环被戳醒，`await q.get()` 立刻返回。

> 记住这条：**从非事件循环线程碰任何 asyncio 对象，一律走 `call_soon_threadsafe`。** 把它包成一个私有方法 `_push(aio_queue, item)`，全项目只留这一个出口，以后 review 一眼就能看出有没有漏网的。

### 5.3 一个前置条件

`self.loop` 必须在**第一个请求进来之前**就注入好（lifespan 里做，`api_server.py:159` 已经写好了）。如果 `add_request` 时 `self.loop is None`，说明有人没走 lifespan 直接构造了引擎 —— 直接抛异常，别静默降级，否则就退化成 §5.1 的诡异 bug。

---

## 6. 主循环：逐行设计 + 时序论证

```python
def _engine_loop(self):
    cursors: dict[int, int] = {}
    try:
        while not self._shutdown:
            # ① 先收割入口队列（新请求 + abort 消息）
            self._drain_incoming()

            # ② 没活儿干 → 挂起，绝不打空调度
            if self.engine.is_finished():
                self.new_request_event.wait()
                self.new_request_event.clear()
                continue

            # ③ 跑一步（阻塞的 CUDA 调用，只在这个线程里发生）
            finished, _ = self.engine.step()

            # ④ 增量 diff + 派发
            self._dispatch_deltas(cursors)
            self._dispatch_finished(finished, cursors)
    except BaseException as e:
        self._exc = e
        self._traceback = traceback.format_exc()
        self._fail_all_requests(e)   # 见 §8.3，不能让 handler 永远挂着
```

### 6.1 为什么必须先判 `is_finished()` 再 `step()`（且顺序不能反）

`Scheduler.schedule()` 的最后一行是 `assert scheduled_seqs`（`kavor/engine/scheduler.py:71`）。`waiting` 和 `running` 都空时调度器什么也排不出来，这个 assert 会直接抛 `AssertionError`。所以第 ② 步是**功能性必需**的，不是优化。

### 6.2 为什么用 `wait()` 而不是 `while True: sleep(0.01)`

空转轮询在 Python 里是灾难：`time.sleep(0.01)` 每秒唤醒 100 次，每次都抢 GIL，不仅白烧一个核心，还会**持续干扰事件循环线程**（GIL 争抢导致 HTTP 延迟抖动）。而且在 §6.1 的前提下你还得额外判 `is_finished()` 才能避免崩溃。

### 6.3 不会丢唤醒（lost wakeup）的论证

经典竞态是："检查条件 → 条件不满足 → 准备睡眠"这三步之间，别的线程把条件改成了满足，然后你才开始睡 —— 你就永远睡了。`threading.Event` 用 `wait()`/`set()` 天然免疫：

> `Event.set()` 之后，**后续任何一次 `wait()` 都立刻返回**，不需要先来后到。

对着时序过一遍：

| 场景 | 事件 | 引擎线程 | 结果 |
|---|---|---|---|
| A | `set()` 发生在 `wait()` **之前** | `wait()` 时 event 已是 set 状态 → 立刻返回 | ✅ 不丢 |
| B | `set()` 发生在 `wait()` **之中** | `wait()` 被 `set()` 唤醒 | ✅ 不丢 |
| C | `set()` 发生在 `clear()` 之后、下一轮 drain 之前 | 下一轮顶部 `_drain_incoming()` 照样能取到（`queue.Queue` 里的数据不会跑），且 event 保持 set，若这轮又空闲则 `wait()` 立刻返回 | ✅ 不丢 |

场景 C 说明**「先 `wait()` 再 `clear()`」这个顺序是对的**：`clear()` 之前必须先经过 `wait()`，不能写成 `clear(); wait()`（那才是真正的丢唤醒）。`continue` 回到循环顶部后重新 drain，所以也不会漏掉那条已经入队的请求。

⚠️ 一个**必须遵守的副作用**：`Event` 只是"叫醒"，它不传递数据，也不保证 `incoming` 里只有一条新消息。所以 `_drain_incoming()` 必须是 **drain 到空**（`while True: try: get_nowait() except queue.Empty: break`），不能只取一条。

### 6.4 一个可以省掉的细节

`step()` 返回的 `num_tokens` 是 prefill/decode 吞吐统计用的，服务化用不上，直接忽略。

---

## 7. 增量输出：游标 diff 的正确性

`LLMEngine.step()` 只在 sequence **结束**时返回结果（`kavor/engine/llm_engine.py:54`），服务化需要每步拿到新 token。**不需要改引擎**，在调用侧维护游标即可：

```python
# cursors: {seq_id: 已经推送出去的 completion token 数}

# ① running 里的序列：按游标切增量
for seq in self.engine.scheduler.running:
    seen = cursors.get(seq.seq_id, 0)
    new = seq.completion_token_ids[seen:]
    if new:
        cursors[seq.seq_id] = seen + len(new)
        self._push(state.queue, ("delta", new))

# ② 刚结束的序列：step() 的返回值里带全量 completion_token_ids，
#    同样按游标切最后一段（第三步：推 finish + 哨兵，并 pop 掉 cursor）
```

逐一验证四种状态都不会出错：

| 情况 | 引擎行为 | 游标 diff 的结果 |
|---|---|---|
| **prefill 中 / chunked prefill 未完成** | `postprocess` 里 `continue`，不 append token（`scheduler.py:86`） | `completion_token_ids` 没变，diff 为空 ✅ |
| **preempt（显存不够被抢占）** | seq 退回 `waiting`，`token_ids` 完整保留（`scheduler.py:75`），块释放 | seq 只是暂时不在 `running` 里，游标不受影响；重新调度后 diff 接着上次的位置继续 ✅ |
| **正常 decode** | 每步 append 一个 token | 立刻 diff 出那一个 token ✅ |
| **刚结束** | `postprocess` 先 append 再置 FINISHED 并从 `running` 移除 | `running` 里已经找不到它 → 由 ② 用 `step()` 返回值兜底，从同一个游标切出最后一段 ✅ |

**先做 ① 再做 ②**，因为结束的 seq 已经不在 `running` 里了，两者不会重复推送同一段 token。

**两个必须处理的收尾**：
- seq 结束时 `cursors.pop(seq_id)`，否则长跑服务会内存泄漏。
- seq 结束时 `tracker.pop(seq_id)`，同理。**顺序**：先把 finish 消息和哨兵推给客户端，再 pop。

### 7.1 finish_reason 判定

不改 scheduler，用现有信息推断。**但注意：两个分支能拿到的数据不一样。**

**① running 分支**（手上是 `Sequence` 对象）：

```python
finish_reason = "length" if seq.num_completion_tokens >= seq.max_tokens else "stop"
```

**② 结束分支**（手上只有 `step()` 返回的 `(seq_id, completion_token_ids)`）：

```python
finish_reason = "length" if len(token_ids) >= state.max_tokens else "stop"
```

⚠️ **别照抄 ① 的写法去写 ②** —— 那段代码在结束分支上根本写不出来：
- `step()` 的返回值里**没有 `Sequence` 对象**，只有 `seq_id` 和 token 列表（`llm_engine.py:54`）；
- 而且这个 seq 已经被 `postprocess` 从 `running` 里移除了（`scheduler.py:92`），waiting 里也没有，**无处反查**。

两个公式是**恒等**的（`len(completion_token_ids) == num_completion_tokens`，见 `sequence.py:44`；`state.max_tokens` 与 `seq.max_tokens` 同源于 `SamplingParams`），差别只在可达性。所以 §4 要求把 `max_tokens` 提前存进 `RequestState`。

另外这个写法有个额外好处：`len(token_ids)` 取的是引擎返回的全量结果，**与"delta 有没有成功推给客户端"无关** —— 就算 `_push` 在关闭期丢了消息（见 §8.2），也不会污染 finish_reason。

注意两个边界：`ignore_eos=True` 时 eos 不参与终止（`scheduler.py:89`），所以只有 max_tokens 这一条路，判成 `"length"` 是对的；eos 和 max_tokens 恰好同时命中时会判成 `"length"`，可接受。引擎内部用的是 `==`（`scheduler.py:89`），我们这里用 `>=`，是有意选的保守写法。

---

## 8. 生命周期：启动 / 关闭 / 异常

### 8.1 启动

```python
def __init__(self, model, **kwargs):
    self.engine = LLMEngine(model, **kwargs)   # ★ 阻塞：加载权重、建 CUDA 上下文、TP 子进程
    ...
    self.thread = threading.Thread(target=self._engine_loop, daemon=True)
    self.thread.start()   # ← 线程在这里才启动
```

`LLMEngine.__init__` 是重阻塞操作（H100 上加载 8B 模型几十秒）。它发生在 **lifespan 的 `yield` 之前**（`api_server.py:151`），也就是"服务还没开始接客"的阶段，阻塞事件循环是**可以接受的**——启动期没有并发请求要伺候。这也是为什么 `AsyncEngine.__init__` 保持同步、不写成 `async def` 的原因之一。

⚠️ 两个别踩的坑：
- **别在模块 import 时构造引擎**。`--reload` / 多 worker 会把引擎复制多份，显存直接翻倍。
- **uvicorn 永远单 worker、禁用 `--reload`**。这条已经写在 `api_server.py:214` 的注释里了。

### 8.2 关闭

`exit()` 会在**事件循环线程**里被调用（lifespan 的 `yield` 之后，`api_server.py:163`）。而 `LLMEngine.exit()` 会 `join()` TP 子进程，是阻塞的。同时引擎线程可能正挂在 `wait()` 上。正确的收尸顺序：

```python
def exit(self):
    self._shutdown = True               # ① 告诉循环别再 step 了
    self.new_request_event.set()        # ② 唤醒可能正睡着的引擎线程
    if self.thread.is_alive():
        self.thread.join(timeout=...)   # ③ 等它跑完当前这一步、退出循环
    self.engine.exit()                  # ④ 线程已经死了，此时从任何线程调用都安全
```

要点：
- **顺序不能反**。先 `join` 再 `engine.exit()`：如果先 `engine.exit()`，引擎线程可能正卡在 `step()` 里，`model_runner.call("exit")` 会和它抢 `dist` / `shm`，报一堆莫名其妙的 NCCL 错误。
- ④ 之所以安全，是因为 ③ 已经保证了引擎线程不再活动。`LLMEngine` 有 `atexit` 兜底（`llm_engine.py:35`），但显式调用更干净可控。
- `join(timeout=...)` 给个上限（比如 10s），别让自己在 Ctrl-C 时永久挂住；超时就打日志说明有任务卡住了。

### 8.2.1 未完成的请求必须推哨兵（而且不能在 `exit()` 里推）

关闭时可能还有 handler 挂在 `await queue.get()` 上。给 `tracker` 里每个请求推 `("finish", "abort")` + `None`，否则这些请求会挂到客户端超时为止。但**从哪里推**有两个坑：

**坑 1：`_push` 在关闭期不能用了。** `_push` 的实现要加两层保护：

```python
def _push(self, aio_queue, item):
    loop = self.loop
    if loop is None or loop.is_closed():
        return                                    # 静默丢弃：此刻已经没有消费者了
    try:
        loop.call_soon_threadsafe(aio_queue.put_nowait, item)
    except RuntimeError:
        pass                                      # is_closed() 检查和真正调用之间仍有窗口
```

- `self.loop is None`：lifespan 还没执行 `engine.loop = ...` 就有人推（`api_server.py:159`）→ 否则 `AttributeError: 'NoneType' object has no attribute 'call_soon_threadsafe'`。
- loop 已关闭：`BaseEventLoop._check_closed()` 抛 `RuntimeError: Event loop is closed`。
- **还有一层更隐蔽的**：`call_soon_threadsafe` 只是**排一个回调**。uvicorn 收尾时 loop 停下，**尚未执行的 ready 回调会被直接丢弃** —— 于是哨兵永远发不出去，请求还是挂死。所以这里"静默丢弃"的语义是对的：丢一个 token 远好于让引擎线程在收尾时抛异常。

**坑 2：`exit()` 本身跑在事件循环线程上**（lifespan 的 `yield` 之后），所以它**不该绕 `_push`**，直接 `queue.put_nowait(item)` 就行 —— 本来就在对的线程里，多绕一层反而撞上坑 1。把这条写成 `exit()` 里的注释。

### 8.2.2 `tracker` 的跨线程访问

引擎线程是 `tracker` 的唯一所有者（drain 时写入、结束时 pop）。`exit()` 在事件循环线程里遍历它推哨兵 —— **如果 `join(timeout=...)` 超时了，引擎线程还活着**，这时候遍历就是对"正在被修改的 dict"做迭代，`RuntimeError: dictionary changed size during iteration` 是必崩的。

两个选择：

- **简单版（推荐第一版）**：只在引擎线程的 `finally` 里做 flush（它是 tracker 的所有者，天然无竞态），`exit()` 里 join 成功就什么都不用做、join 超时就打一条警告日志走人。
- **完整版**：加一把 `threading.Lock` **只**包住 dict 的读/写/迭代 —— **绝不**包住 `step()` / `add_request()` / `join()` / `call_soon_threadsafe`（那样等于把整个引擎串行化，还会死锁）。用锁的纪律比锁本身更重要。

### 8.2.3 Ctrl+C：TP 子进程和前台进程组

这是最容易漏的一环。**Ctrl-C 不是只发给父进程** —— 终端把 SIGINT 发给整个**前台进程组**，`mp.get_context("spawn")` 起的 TP 子进程也在里面（multiprocessing 默认不新建 session/进程组）。

于是默认行为是：父进程收到 SIGINT 开始优雅退出，子进程同时也收到 SIGINT **当场死掉**。然后父进程走进 `ModelRunner.exit()` 里的 `dist.barrier()` —— 对端已经没了 —— **永久挂死**。表现就是「Ctrl+C 后程序卡住，显存不释放，只能 kill -9」。

三道修复，缺一不可：

1. **子进程忽略 SIGINT**（`llm_engine.py:32`）。子进程入口第一件事就是 `signal.signal(signal.SIGINT, signal.SIG_IGN)`，退出统一由父进程编排：父进程写 shm + set event → 子进程跳出 `loop()` → 两边都走完 `dist.barrier()` → 父进程 join。注意 spawn 走的是 `exec`，**exec 会把已处理的信号重置成默认值**，所以这个 SIG_IGN 必须在子进程里重新设一次，父进程设了没用。
2. **回收要有三级兜底**：等 → `SIGTERM` → `SIGKILL`。只写 `p.join()` 的话，任何子进程不响应就是永久挂死。
3. **已经有人先死就别走优雅路径**。`exit()` 里先查 `all(p.is_alive())`；有死人就跳过 `model_runner.call("exit")`（那条路必经 `dist.barrier()`，必挂），直接强杀。

另外两个必须处理的时序：

- **`atexit` 要注册在慢速模型加载之前**。原来注册在 `__init__` 最后一行，加载途中被 Ctrl-C 打断时前面已经起好的子进程没人收尸 → 孤儿进程 + 显存不释放。现在注册提前，并且用 `try/except BaseException: self.exit(); raise` 兜住加载失败。
- **`process.daemon = True`**（必须在 `start()` 之前设）。父进程意外退出时解释器会 `terminate()` 这些 daemon 子进程，是最后一道「不留孤儿」的保险。

实测：一个 spawn 子进程从收到退出信号到被 join 回收约 1.3s（空壳进程；真带上 CUDA 上下文会更久），所以 `WORKER_EXIT_TIMEOUT = 10s` 是合适的量级。

**一个已知残留风险**：`kill -9` 父进程时 `/dev/shm/kavor` 不会被 `unlink`，下次启动 `SharedMemory(name="kavor", create=True)` 会因为 `FileExistsError` 起不来，需要手动 `rm /dev/shm/kavor`。正常的 Ctrl+C 路径不会走到这里。

### 8.2.4 `LLMEngine.exit()` 需要幂等 guard

`LLMEngine.__init__` 里有 `atexit.register(self.exit)`（`llm_engine.py:35`），而 lifespan 关闭时也会经 `AsyncEngine` 调一次。第二次进来时 `self.model_runner` 已经被 `del` 掉了（`llm_engine.py:39`），`self.model_runner.call("exit")` 直接 `AttributeError`；就算没 del，`dist.destroy_process_group()` / `p.join()` 再跑一遍也会报 NCCL / 进程错误。

表现是进程退出时打印 `Exception ignored in atexit callback` —— **不改退出码，但污染日志、并且会掩盖真正的关闭错误**，很难查。两行修掉：

```python
def exit(self):
    if getattr(self, "_exiting", False):
        return
    self._exiting = True
    ...
```

### 8.3 引擎线程异常

引擎线程里的异常**不会自动传播到主线程**，如果你不处理，Python 只会打一行 stderr 然后线程静默死亡 —— 而 HTTP 侧还在正常返回 200 的 `/health`。必须：

1. `_engine_loop` 整体包 `try/except BaseException`，用 `traceback.format_exc()` 记下完整栈（存成实例字段或打日志）。
2. 遍历 `tracker`，向每个请求的队列推 `("finish", "error")` + `None`，让在等的请求立刻失败而不是挂死。
3. `is_alive()` 返回 `self.thread.is_alive() and self._exc is None`，这样 `/health`（`api_server.py:167`）能返回 503。
4. 顺手把 `new_request_event` 后续的 `add_request` 直接拒绝掉（`self._exc is not None` 时立刻抛异常），别让新请求石沉大海。

---

## 9. 其他容易忽略的点

### 9.1 请求在消费者开始 await 之前就产 token，会丢吗？

不会。`asyncio.Queue` 是**有缓冲的**：`put_nowait` 只是往内部 deque 里塞，不要求有人正在等。`add_request` 返回队列之后、handler 开始 `await q.get()` 之前引擎线程推的数据，全都老老实实躺在队列里，第一次 `get()` 就能拿到。所以 `add_request` 里**不要**加任何"等消费者就绪"的同步，那是多余的复杂度。

### 9.2 短请求可能一两步就跑完

`max_tokens=1` 的请求可能一两步就跑完。关键是：**`seq_id = self.engine.add_request(...)` 返回后必须立刻登记 `tracker` 条目** —— 不要在这两句中间插任何 `await` 或 IO。

之所以这样就够了：`add_request` 和登记都在引擎线程里**连续**执行，中间不可能插入 `step()`，所以不存在"token 已经产出、但还没登记"的窗口。反过来说，如果你把提交和登记拆到两个线程（比如在 handler 里先登记再入队），那个窗口就真实存在了。

这也是为什么建议 `LLMEngine.add_request` 改成返回 `seq.seq_id`（`llm_engine.py:43`）—— 你只有在引擎线程里调用它的那一次机会能拿到 seq_id。

好消息：改它是**完全向后兼容**的。`add_request` 目前只有 `LLMEngine.generate()` 内部在用（`llm_engine.py:70`），而那处忽略了返回值。

### 9.3 CUDA 隐患：图在主线程 capture，在引擎线程 replay

`ModelRunner.__init__`（主线程）会 `capture_cudagraph()`（`model_runner.py:45`），而之后的 `step()` replay 发生在引擎线程。这是目前设计里**唯一一个没法静态确认的点**：

- 乐观的一面：`torch.cuda.graph` 内部用**自己创建的专属 stream** 来 capture，replay 也是走图内部记录的 stream，不依赖"当前线程的当前 stream"，跨线程 replay 通常是安全的。vLLM V0 的异步引擎也是"主线程加载 + 后台线程执行"这个结构，在生产里跑了很多年。
- 验证方法：先用 `enforce_eager=True`（关闭 CUDA graph，`examples/example.py:15` 已经在用）跑通全流程，再切到默认模式（graph 开启）对比。如果出现 CUDA illegal memory access / 输出乱码 / 结果与 `generate()` 不一致，而 `enforce_eager=True` 正常，那就是这里的问题 —— 兜底方案是在引擎线程里重新 capture，或者短期先常驻 eager 模式。
- 顺带一提：**CUDA 的 current device 是线程局部的**，`ModelRunner.__init__` 里那句 `torch.cuda.set_device(rank)`（`model_runner.py:28`）只对**构造它的线程**生效。引擎线程从没设过，靠的是"新线程默认 device 0"。rank 0 恰好就是 device 0，所以现在没事 —— 但这是**侥幸**，不是设计。建议在 `_engine_loop` 开头显式补一句 `torch.cuda.set_device(self.engine.model_runner.rank)`，成本为零，将来引擎线程要管别的 rank 时不会踩雷。

### 9.4 一个既有 bug：`is_finished()` 为 False 时 `step()` 仍可能 assert

§6.1 说"先判 `is_finished()` 就不会撞 assert" —— 这话**不完全对**。看 `schedule()` 的实际控制流（`scheduler.py:30-71`）：

- prefill 循环里，`can_allocate(seq)` 返回 `-1`（free block 不够）时直接 `break`；
- 此时若 `scheduled_seqs` 为空，`if scheduled_seqs:` 不成立，跳过去看 decode 循环；
- decode 循环条件是 `while self.running`，running 为空 → 一次都不进；
- 最后 `assert scheduled_seqs` → **崩**。

触发条件：**waiting 队头那条 seq 需要的 block 数超过整个 KV cache 的总容量**（prompt 特别长，或 `gpu_memory_utilization` 给小了导致 `num_kvcache_blocks` 很小）。此时 `is_finished()` 返回 `False`（waiting 非空），但 `step()` 必炸。

**为什么在服务里这条特别要命**：同步 CLI 里一个坏 prompt 只是让这一次 `generate()` 崩掉，你重跑就行；在服务里，它会让**引擎线程直接死掉，连累所有在途请求**（拿不到哨兵），`/health` 变 503 —— 一个用户的超长请求打挂整个服务。

两道防线：
1. **API 层校验**：请求进来就检查 `prompt_tokens + max_tokens <= max_model_len`，超了直接 400。注意这只挡住了"超过 `max_model_len`"的情况，挡不住"`max_model_len` 本身大于 KV cache 总容量"的配置 —— 后者要靠 2。
2. **引擎线程异常必须终止而不是重试**：`_engine_loop` 的 `try` 要包住整个 `while` 循环，异常后**退出循环**（§8.3）。如果写成 `except: continue`，线程会不停重撞同一个 assert 空转刷日志。

### 9.5 `torch.multiprocessing` 的 spawn 与线程

`LLMEngine.__init__` 里 `mp.get_context("spawn")` 起的 TP 子进程是在**主线程**完成的，早于引擎线程 start。spawn 进程需要重新 import 主模块，别让子进程在 import 期碰到 `AsyncEngine`（比如写在模块顶层的构造逻辑），否则会递归起线程。

---

## 10. abort 设计

### 10.1 触发点

| 触发方式 | 检测位置 |
|---|---|
| 客户端断开 SSE | 流式生成器里 `await request.is_disconnected()`，或 `asyncio.CancelledError`（`StreamingResponse` 在客户端断开时会 cancel 生成器） |
| 显式取消接口 | `POST /abort` 或 `DELETE /v1/.../{request_id}` |
| 超时 | 服务端自己设 deadline |

推荐**先只做流式断开这一条**：它在 `stream_sse` 那个 async generator 里，`finally: engine.abort(request_id)` 一行就能盖住"断开"和"正常结束"两种情况（正常结束时 abort 应当是幂等的 no-op）。

### 10.2 核心设计决策：abort 必须是**消息**，不是共享标志位

直觉做法是 `engine.aborted.add(request_id)` 让引擎线程自己去查。**不要这么做**，两个理由：

1. **线程边界**：反正 marker 也得加锁，不如直接复用已有的 `incoming` 队列（它已经是唯一的跨线程入口），少一个需要论证正确性的同步原语。
2. **更重要的：它天然消灭了"正在被调度中"的竞态。** 引擎线程是严格串行的：`drain → step → dispatch → drain → step → ...`。abort 消息**只可能在两次 `step()` 之间被处理**（循环顶部的 drain）。也就是说，不存在"seq 已经被放进 `scheduled_seqs`、CUDA 正在跑，这时候有人把它从 `running` 里抽走"的窗口 —— 而这个窗口恰恰是最难处理的（`postprocess` 末尾会 `self.running.remove(seq)` 再抛 `ValueError`）。

如果不走队列而走共享标志位，你就必须自己论证这个窗口，或者改 `postprocess` 加容错，复杂度立刻上一个台阶。

### 10.3 `Scheduler.abort(seq_id)`

```python
def abort(self, seq_id):
    for dq in (self.waiting, self.running):
        for i, seq in enumerate(dq):     # 别直接 for seq in dq 然后 remove，会跳过元素
            if seq.seq_id == seq_id:
                del dq[i]
                self.block_manager.deallocate(seq)
                seq.status = SequenceStatus.FINISHED
                return True
    return False
```

要点：
- **`deallocate` 对"还没分配过 block"的 seq 是安全的**：`block_table` 为空时循环体不执行（`block_manager.py:94`）。所以"刚 add 进来还没 prefill 就被 abort"这种最常见的场景不会炸。
- **不要**从 `running` 里移除后就以为完事了：**必须 `deallocate`**，否则 KV cache block 的 `ref_count` 永远不归零，几轮下来就把显存耗光 —— 这是 abort 最容易漏的一步。
- 遍历时用索引删除，或者 `list(dq)` 复制一份再操作，别在遍历中改 `deque`。

### 10.4 引擎线程侧要处理三种"还没被消费"的情况

1. **请求还在 `incoming` 里没被 drain**（很短但真实的窗口）：drain 时先查一个 `self.aborted_request_ids: set`，命中就直接推哨兵、不调 `engine.add_request()`。所以 **abort 的 set 要由事件循环线程写、引擎线程读**（`set.add`/`in` 在 CPython 下是原子的，且这里的语义只需最终一致，不必加锁 —— 但要在注释里写清楚这个判断）。
2. **请求在 `waiting` 里**（还没 prefill，或已被 preempt）：`Scheduler.abort()` 能直接删掉。
3. **请求在 `running` 里**：同上。

统一收尾：`abort` 成功后推 `("finish", "abort")` + `None`，`tracker.pop`、`cursors.pop`。

### 10.5 request_id → seq_id 的反查

`Scheduler.abort()` 需要 `seq_id`，而 HTTP 侧只知道 `request_id`。所以要维护 `self.request_id_to_seq: dict[str, int]`（在 drain 时填，在结束时 pop）。反查不到 = 请求已经结束或还没被消费，此时：
- 若在 `aborted_request_ids` 里没记录过 → 记上（覆盖情况 1）；
- 无论如何都**幂等**返回，不报错。

---

## 11. 坑清单（对照 `fastapi_serving.md` 第 8 节，补三条）

| 坑 | 说明 |
|---|---|
| 空调度崩溃 | 必须先 `is_finished()` 再 `step()`，`scheduler.py:71` 的 assert 会炸 |
| 跨线程碰 scheduler | 无锁。`add_request`/`step`/`abort` 只能在引擎线程调 |
| handler 里直接调 `step()` | 阻塞 CUDA 调用会冻住整个事件循环 |
| 忘记 `call_soon_threadsafe` | 数据进去了但消费端不醒（§5.1）—— 表现为随机的、几秒到几十秒的首 token 延迟 |
| `wait()`/`clear()` 顺序写反 | `clear(); wait()` 会丢唤醒，然后引擎线程永久卡死（§6.3） |
| `drain` 只取一条 | `Event` 不传数据，必须 drain 到空 |
| 引擎线程异常没兜住 | 线程静默死亡，`/health` 还在报 200（§8.3） |
| 关闭顺序反了 | 先 `engine.exit()` 再 `join` 会和 `step()` 抢 `dist`/`shm`，报 NCCL 错 |
| 关闭时不清队列 | handler 挂在 `await q.get()` 直到客户端超时 |
| 关闭期还用 `_push` 推哨兵 | loop 已关闭会抛 `RuntimeError`；就算不抛，排进回调队列但未执行也会被丢弃 → 请求还是挂死（§8.2.1） |
| join 超时后遍历 `tracker` | `RuntimeError: dictionary changed size during iteration`（§8.2.2） |
| `LLMEngine.exit()` 不幂等 | `atexit` + lifespan 双调用 → `AttributeError` / NCCL 报错（§8.2.4） |
| TP 子进程跟着 Ctrl+C 一起死 | 父进程的 `dist.barrier()` 永久挂死，显存不释放（§8.2.3） |
| teardown 没有 terminate/kill 兜底 | 子进程不响应就永久挂死（§8.2.3） |
| 超长 prompt | 会撞 `assert scheduled_seqs`，在服务里打死整个引擎线程（§9.4） |
| `cursors` / `tracker` 不 pop | 长跑内存泄漏 |
| abort 忘了 `deallocate` | KV cache block 泄漏，最终 OOM |
| `temperature=0` | `SamplingParams` 断言失败，服务端映射成 `1e-6` |
| `max_tokens` 默认 64 | 不显式传就只有 64 个 token |
| decode 单个 token | 多 byte UTF-8 会被切碎，用"全量 decode + diff"（`api_server.py:139` 已处理） |
| `--reload` / 多 worker | 引擎和 TP 子进程被复制多份 |

---

## 12. 验证清单（按这个顺序做，每步独立可验证）

**第 0 步 · 无 GPU 验证线程模型**
先写一个**假 `LLMEngine`**（`step()` 里 `time.sleep(0.05)`，手工往 `Sequence` 上 append token）来跑 `AsyncEngine`。这样能安全地压测竞态、关闭、异常路径，不用等几十秒加载模型。

- `asyncio.run()` 里并发 `add_request` 5 个，逐个消费到哨兵 → 输出正确
- **交错完成**：先来一个长请求，中途插入短请求，短请求先结束且不影响长请求
- `finish_reason` 三态：`ignore_eos=True, max_tokens=N` → `"length"`；正常 eos → `"stop"`
- 引擎线程空闲时 `wait()` 挂起（用 `py-spy dump` 或打印确认没有空转）
- **丢唤醒压测**：`asyncio.gather` 起 200 个并发 `add_request`，全部收齐哨兵，且最后 `tracker` / `cursors` 全部为空（漏 pop 会在这里暴露）
- 处理中调 `exit()`，所有 handler 立刻拿到哨兵而不是挂死
- **幂等**：连续调两次 `exit()` 不抛异常
- 故意让 `step()` 抛异常 → `/health` 返回 503，在等的请求立刻失败；之后再 `add_request` 应被明确拒绝（而不是石沉大海）

**第 1 步 · 增量输出对拍（真实引擎）**
`enforce_eager=True`，`add_request` 两个 prompt，循环 `step()` 打印每步 diff。断言：**把每步 diff 拼起来 == `LLMEngine.generate()` 的最终 `token_ids`**。这一步能一次性验证 §7 表格里的四种状态。

**第 2 步 · 服务链路**

```bash
kavor serve /path/to/model --tp 1 --port 8000
curl http://localhost:8000/health
curl http://localhost:8000/generate -H 'Content-Type: application/json' \
  -d '{"request_id":"t1","prompt":"introduce yourself","max_tokens":100}'
curl -N http://localhost:8000/generate -H 'Content-Type: application/json' \
  -d '{"request_id":"s1","prompt":"写一首诗","stream":true,"max_tokens":200}'
```

**第 3 步 · 并发与连续批处理**（验收清单里的核心项）
- 两个终端同时 curl：观察是否**交错**输出（continuous batching 生效，而非串行）
- 长请求进行中插入短请求：短请求先完成，且不影响长请求
- `stream=true` 与 `stream=false` 输出一致
- 中文流式无乱码（§5.1 那个 bug 在这里会以"字卡住不吐"的形式暴露）

**第 4 步 · 稳定性**
- Ctrl-C 后 `ps` / `nvidia-smi` 确认 **TP 子进程无残留**、显存释放（TP=2 起一次更有意义）
- 反复起停 3 次，确认日志里没有 `Event loop is closed` / `AttributeError: model_runner` / `Exception ignored in atexit callback` 这类水印 → 验证 §8.2 的关闭路径确实是幂等的
- 客户端流式中途 `Ctrl-C` 断开（abort 做完后）→ 服务端不崩，`nvidia-smi` 显存不涨
- 关掉 `enforce_eager`（开启 CUDA graph）重跑第 3 步 → 验证 §9.3 的隐患是否真实存在

**CUDA 问题怎么定位（§9.3 真出问题时的手段）**
- `CUDA_LAUNCH_BLOCKING=1` 起服务：把异步执行的 kernel 错误变成同步报错，能直接定位到是哪一步炸的。**注意它会让性能暴跌，只用于定位。**
- 同时扫 `dmesg` / 宿主机日志有没有 Xid 报错；有 Xid 基本就坐实了是 CUDA 上下文/图的问题，退 `--enforce-eager` 并考虑"把 `capture_cudagraph` 搬到引擎线程里做"这个 Plan B。

---

## 13. 建议的落地顺序

1. 改 `LLMEngine.add_request` 返回 `seq.seq_id`（一行，向后兼容）
2. 写 §7 的游标 diff，用假引擎对拍验证
3. 写 `AsyncEngine` 骨架：数据结构 + 主循环 + `_push`（先不带 abort）
4. 用假 `LLMEngine` 跑第 0 步的全部用例
5. 接真实引擎跑第 1、2 步
6. `api_server.py` 的 lifespan 切到 `AsyncEngine`（`api_server.py:79` 的 `build_app`）
7. 跑第 3、4 步
8. 全部通过后再加 abort（§10）

---

## 14. 附录：`fastapi_serving.md` §3 骨架的勘误

那份文档第 3 节的骨架是**示意性伪代码**，照着直接敲会卡住或埋雷。逐条对照（每条给出本文的对应小节）：

| # | 位置 | 问题 | 见 |
|---|---|---|---|
| 1 | L143 `while True:` | 没有任何 break 条件，类里也没有 `exit()`，而 `api_server.py:163` 要调 `engine.exit()` → lifespan 关闭时 `AttributeError`；补个空 `exit()` 也没用，线程永远挂在 `wait()` 上 | §8.2 |
| 2 | L122-132 | 缺 `is_alive()` 和 `tokenizer`，但 `api_server.py:170/195/202` 都要用 → `/health` 一调就 500 | §3 |
| 3 | L152 | `seq_id` 是未定义变量（L149 没接 `add_request` 的返回值），`RequestState(...)` 也是占位 → `NameError`，且 `tracker` 建不起来，请求的队列永远没有消费者 | §9.2 |
| 4 | L105 | `finish_reason` 用 `seq.num_completion_tokens` 判定，但结束分支拿不到 `seq` 对象，也反查不到 | §7.1 |
| 5 | L142/166-168 | `cursors` 的归属、两个 `_dispatch` 的接口都没定义，且 §2 的 `diff_new_tokens(engine, cursors)` 签名和 §3 的调用形式对不上 → 容易写成两处各推一次、尾部 token 重复 | §7 |
| 6 | L170-173 | `_push` 没有空 loop / 已关闭 loop 的保护；且 `call_soon_threadsafe` 只是排回调，关闭期会被丢弃 | §8.2.1 |
| 7 | L143-169 | `_engine_loop` 没有 `try/except` → 引擎线程静默死亡，所有在途请求拿不到哨兵 | §8.3 |
| 8 | L134-139 | `add_request` 不检查引擎存活性 → 线程死后新请求入队即石沉大海 | §8.3 |
| 9 | L128/152 | `tracker` 的跨线程访问没有约定（`exit()` 遍历 vs 引擎线程 pop） | §8.2.2 |
| 10 | L147-148 | `queue.empty()` + `get()` **不是原子对**，`empty()` 文档明确说不保证。单消费者下能用，但语义不清晰 | §6.3 |
| 11 | L129-130 | `RequestState` 里的 `prompt_token_ids` 拿不到（`add_request` 内部才 tokenize 且不返回），`prompt` 全文常驻也浪费 | §4 |
| 12 | L155-156 | 三个边界没提：CUDA current device 是线程局部的、`LLMEngine.exit()` 需要幂等 guard、`is_finished()` 为 False 时 `step()` 仍可能 assert | §9.3 / §8.2.4 / §9.4 |
| 13 | L145/171 | `_push` 的注释容易被读成"直接 put、线程安全"，应写明它只负责排回调，真正执行 `put_nowait` 的是 loop 线程 | §5.2 |

**另外一处不属于 §3、但实现时会撞上的**：§4 骨架 L232 的 `tokens.extend(item)` 假设队列里是裸的 token 列表，和 §3 的 `("delta", list[int])` 元组协议**冲突**。以 `api_server.py` 里 `consume_all()` / `stream_sse()` 的协议为准（`("delta", ...)` / `("finish", ...)` / `None`）。

**一条隐含假设，值得写明**：`stream_sse` 里的 `tokenizer.decode()` 跑在事件循环线程，而引擎线程会调 `tokenizer.encode()`。HuggingFace fast tokenizer 内部有锁，实践上可行 —— 但这是个**假设**，不是文档保证，心里有数就行。
