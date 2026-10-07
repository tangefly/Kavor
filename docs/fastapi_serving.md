# Kavor FastAPI 服务化操作指南

本文档描述如何给 Kavor 加一层类似 vLLM 的 OpenAI 兼容 FastAPI 服务。按步骤顺序执行，每一步都有独立验证方法。

---

## 0. 总体架构

Kavor 引擎层（`Scheduler` / `LLMEngine`）已经具备 continuous batching、chunked prefill、preemption，**不需要大改**。要做的是把 `LLMEngine.generate()` 这种"整批阻塞到全部完成"的同步接口，包装成服务化需要的"增量输出 + 异步并发"：

```
┌─────────────────────────────────────────────────────────┐
│  FastAPI (HTTP / SSE)                                   │
│    /v1/chat/completions  /v1/completions  /health ...   │
└──────────────────────┬──────────────────────────────────┘
                       │ 每请求一个 asyncio.Queue
┌──────────────────────▼──────────────────────────────────┐
│  AsyncEngine（后台 daemon 线程）                          │
│   - 引擎主循环：有请求 → step()，无请求 → Event 阻塞      │
│   - RequestTracker：seq_id → queue / 元数据              │
│   - call_soon_threadsafe 把增量 token 推回 asyncio 世界  │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  LLMEngine（现有代码，基本不动）                          │
│   Scheduler / ModelRunner / BlockManager                │
└─────────────────────────────────────────────────────────┘
```

新增文件布局（照抄 vLLM 的组织方式）：

```
kavor/
  engine/
    async_engine.py          # 第二步：异步封装层
  entrypoints/
    __init__.py
    openai/
      __init__.py
      api_server.py          # 第三步：FastAPI app + CLI 入口
      protocol.py            # 第五步：OpenAI 协议 Pydantic 模型
      serving_chat.py        # 第五步：/v1/chat/completions
      serving_completion.py  # 第五步：/v1/completions
```

**实施顺序**（每步可独立验证）：

1. 增量输出（游标 diff，不改引擎）
2. AsyncEngine 封装层
3. FastAPI 骨架 + 裸 `/generate` 调试端点（非流式）
4. SSE 流式
5. OpenAI 兼容协议层
6. 收尾：usage 统计、health、CLI、（可选）abort

---

## 1. 前置准备

- 确认依赖：`fastapi`、`uvicorn`。SSE 不需要额外库（用 `starlette.responses.StreamingResponse` 即可）。
- **`Config` 要求 `model` 必须是本地目录**（`kavor/config.py` 里有 `assert os.path.isdir(self.model)`），所以服务的 `--model` 参数只接受本地路径，不接受 HF hub id。
- 默认 `max_model_len=4096`、`max_num_seqs=512`、`max_num_batched_tokens=16384`，协议层做长度校验时会用到这几个值。

---

## 2. 第一步：增量输出（游标 diff，不改引擎）

### 目标

`LLMEngine.step()`（`kavor/engine/llm_engine.py:49`）只在 sequence 结束时返回结果。服务化需要每步拿到新产生的 token。

### 方法

**不要改引擎**。在调用侧维护一个游标字典：

```python
# {seq_id: 该请求已经推送给客户端的 completion token 数}
cursors: dict[int, int] = {}

def diff_new_tokens(engine, cursors):
    """每步 step() 之后调用，返回 {seq_id: 新增的 token 列表}"""
    deltas = {}
    # 1) 还在跑的请求：遍历 scheduler.running，按 len(completion_token_ids) 切增量
    for seq in engine.scheduler.running:
        seen = cursors.get(seq.seq_id, 0)
        new = seq.completion_token_ids[seen:]
        if new:
            cursors[seq.seq_id] = seen + len(new)
            deltas[seq.seq_id] = new
    # 2) 刚结束的请求：step() 返回值里有完整 completion_token_ids，
    #    同样按游标切出最后一段增量，然后清理游标
    #    （在 AsyncEngine 主循环里结合 step() 的返回值一起做）
    return deltas
```

**为什么安全**：
- preemption 只是把 seq 移回 waiting 并释放 block，`token_ids` 完整保留，游标 diff 不受影响；
- chunked prefill 未完成的步里 `postprocess` 会 `continue`，不 append token，diff 自然为空；
- 结束的 seq 会从 `running` 移除，由 `step()` 返回值兜底（拿全量 `completion_token_ids` 再按游标切最后一段）。

### finish_reason 判定

OpenAI 协议需要区分结束原因。不需要改 scheduler，用现有信息推断：

```python
finish_reason = "length" if seq.num_completion_tokens >= seq.max_tokens else "stop"
```

### 验证

写个临时脚本：`add_request` 两个 prompt，循环 `step()`，每步打印 diff 出的 token，确认与 `generate()` 的最终输出一致、且 token 是逐个/逐批出现的。

---

## 3. 第二步：AsyncEngine 封装层（`kavor/engine/async_engine.py`）

### 目标

`step()` 是阻塞的 GPU 调用，**绝不能直接跑在 asyncio 事件循环里**（会卡死所有 HTTP 处理）。用一个后台线程跑引擎主循环，通过队列与 asyncio 通信。

### 核心结构

```python
class AsyncEngine:
    def __init__(self, model, **kwargs):
        self.engine = LLMEngine(model, **kwargs)
        self.loop = None                    # 主事件循环引用，lifespan 里注入
        self.new_request_event = threading.Event()
        self.incoming = queue.Queue()       # 线程安全的入口队列
        self.tracker = {}                   # seq_id -> RequestState
        # RequestState: asyncio.Queue, request_id, prompt, sampling_params,
        #               finish_reason, prompt_token_ids, arrival_time
        self.thread = threading.Thread(target=self._engine_loop, daemon=True)
        self.thread.start()

    async def add_request(self, request_id, prompt, sampling_params) -> asyncio.Queue:
        """HTTP handler 调这个（async 上下文）。返回该请求的输出队列。"""
        q = asyncio.Queue()
        self.incoming.put((request_id, prompt, sampling_params, q))
        self.new_request_event.set()        # 唤醒引擎线程
        return q

    def _engine_loop(self):
        cursors = {}
        while True:
            # 1) 先收割入口队列里的新请求
            #    注意：add_request 里带 token 化（str → token ids）由
            #    LLMEngine.add_request 在本线程内完成，保证 scheduler 线程安全
            while not self.incoming.empty():
                request_id, prompt, sp, q = self.incoming.get()
                self.engine.add_request(prompt, sp)   # 内部会 tokenize
                # 记录 seq_id ↔ request_id 的映射（add_request 需要改成返回 seq_id，
                # 或者在 add 之前自己 tokenize 拿到 seq_id —— 二选一）
                self.tracker[seq_id] = RequestState(...)

            # 2) 没有请求时阻塞等待，不要空转
            #    ⚠️ 绝不能在 is_finished() 为 True 时调 step()：
            #    scheduler.schedule() 末尾有 assert scheduled_seqs，会直接崩
            if self.engine.is_finished():
                self.new_request_event.wait()
                self.new_request_event.clear()
                continue

            # 3) 执行一步
            finished, _ = self.engine.step()

            # 4) 游标 diff 出增量，推给对应请求的队列
            deltas = diff_new_tokens(...)          # 第一步写好的函数
            self._dispatch(deltas, cursors)
            self._dispatch_finished(finished, cursors)   # 推 finish_reason + None 哨兵

    def _push(self, aio_queue, item):
        """跨线程往 asyncio.Queue 塞数据的唯一通道"""
        self.loop.call_soon_threadsafe(aio_queue.put_nowait, item)
```

### 关键点

1. **`seq_id` 关联**：`LLMEngine.add_request()`（`kavor/engine/llm_engine.py:43`）现在不返回 seq_id。最简单的改法是让它返回 `seq.seq_id`（一行改动，不影响现有调用）；或者 `AsyncEngine` 在入队前自己 `tokenizer.encode` 再构造 `Sequence`。推荐前者。
2. **线程边界**：`engine.add_request` / `engine.step` 只允许引擎线程调用；HTTP handler 一律走 `incoming` 队列。scheduler 没有锁，跨线程直接调会坏。
3. **`self.loop` 注入**：FastAPI lifespan 里 `engine.loop = asyncio.get_running_loop()`。`call_soon_threadsafe` 必须用主 loop 的引用，从别的线程拿不到。
4. **结束哨兵**：请求结束时向队列推 `(finish_reason, ...)` 后再推一个 `None`，消费端的 async generator 遇到 `None` 退出。
5. **chat template 的 tokenizer 使用**：`apply_chat_template` 是纯 Python 字符串操作，放在 HTTP handler 里做是安全的；真正的 `encode` 留给引擎线程（`add_request` 内部完成）。

### 验证

写个临时脚本：`asyncio.run()` 里起 AsyncEngine，并发 `add_request` 若干个，逐个消费队列直到哨兵，检查输出与离线 `generate()` 一致；再验证"先来一个长请求、中途再来一个短请求"能正常交错完成（continuous batching 生效）。

---

## 4. 第三步：FastAPI 骨架 + 裸调试端点（`kavor/entrypoints/openai/api_server.py`）

### 目标

先跑通"HTTP → AsyncEngine → 响应"链路，协议格式后面再套。

### 结构

```python
# api_server.py
import argparse
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)   # 只接受本地路径
parser.add_argument("--tp", "--tensor-parallel-size", type=int, default=1)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
# 其它 Config 字段按需透传（enforce_eager、max_model_len、max_num_seqs ...）

@asynccontextmanager
async def lifespan(app):
    engine = AsyncEngine(args.model, tensor_parallel_size=args.tp, enforce_eager=True)
    engine.loop = asyncio.get_running_loop()
    app.state.engine = engine
    yield
    engine.exit()          # 触发 LLMEngine.exit()，join TP 子进程

app = FastAPI(lifespan=lifespan)

@app.post("/generate")
async def generate(raw: dict):
    q = await app.state.engine.add_request(
        request_id=raw["request_id"], prompt=raw["prompt"],
        sampling_params=SamplingParams(temperature=..., max_tokens=...))
    tokens = []
    while True:
        item = await q.get()
        if item is None:
            break
        tokens.extend(item)          # item = token id 列表
    return {"text": tokenizer.decode(tokens)}

if __name__ == "__main__":
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
```

### 关键点

1. **引擎必须在 lifespan 里创建**，不能在模块 import 时创建（`--reload` / 多 worker 会各复制一份引擎 + TP 子进程）。
2. **uvicorn 只能单 worker、禁用 `--reload`**。
3. 想注册成命令行工具的话，在 `pyproject.toml` 加：
   ```toml
   [project.scripts]
   kavor-serve = "kavor.entrypoints.openai.api_server:main"
   ```
   （需要把 argparse 逻辑抽到 `main()` 里）

### 验证

```bash
python -m kavor.entrypoints.openai.api_server --model /path/to/Qwen3-8B --tp 1 --port 8000
curl http://localhost:8000/generate -H 'Content-Type: application/json' \
  -d '{"request_id": "test-1", "prompt": "introduce yourself", "max_tokens": 100}'
```

另开两个终端同时 curl，观察服务端日志确认两个请求被连续调度（而非串行处理完一个再收下一个）。

---

## 5. 第四步：SSE 流式

### 方法

给 `/generate` 加 `stream` 参数，流式时返回 `StreamingResponse`：

```python
from fastapi.responses import StreamingResponse

async def _stream_generator(q):
    while True:
        item = await q.get()
        if item is None:
            yield "data: [DONE]\n\n"
            return
        yield f"data: {json.dumps({'text': decode(item)})}\n\n"

if raw.get("stream"):
    return StreamingResponse(_stream_generator(q), media_type="text/event-stream")
```

### 增量 detokenize

第一版用"全量 decode + 文本 diff"，简单且正确：

```python
# 每次收到新 token 后：
new_text = tokenizer.decode(all_completion_token_ids, skip_special_tokens=True)
delta = new_text[len(sent_text):]
sent_text = new_text
```

- O(n²)，但先跑通；后面有需要再升级成 vLLM 式增量 detokenizer（要处理多 byte UTF-8 字符被切在两个 token 里的缓冲问题）。
- 不要直接 `tokenizer.decode(delta_ids)` —— 会把一个中文字符的两个 token 解出乱码。

### 验证

```bash
curl -N http://localhost:8000/generate -H 'Content-Type: application/json' \
  -d '{"request_id": "s-1", "prompt": "写一首诗", "stream": true, "max_tokens": 200}'
```

`-N` 关闭缓冲，应能看到 data: 一段段逐步出现；中文不乱码。

---

## 6. 第五步：OpenAI 兼容协议层

### protocol.py — Pydantic 模型

定义（字段对齐 OpenAI 规范）：

- 请求：`ChatCompletionRequest`（messages, model, temperature, max_tokens / max_completion_tokens, stream, stream_options）、`CompletionRequest`（prompt, ...）
- 响应：`ChatCompletionResponse`（id / object / created / model / choices[{index, message{role, content}, finish_reason}] / usage）、流式的 `chunk` 变体、`CompletionResponse`
- `model` 字段校验放宽：客户端经常传模型名，服务端忽略或与实际模型比对后返回 400，二选一。

参数映射（有坑，逐条对照 `kavor/sampling_params.py`）：

| OpenAI 参数 | Kavor | 注意 |
|---|---|---|
| `temperature=0`（greedy） | **不能直接传** | `SamplingParams` 断言 `temperature > 1e-10`，映射成 `1e-6` 之类的 epsilon |
| `max_tokens` / `max_completion_tokens` | `max_tokens` | 引擎默认值只有 64，服务端必须显式给：请求值或默认值，且 ≤ `max_model_len - prompt_tokens` |
| `stop` / `n>1` / `logprobs` / `top_p` 等 | 不支持 | 第一版直接返回 400（带明确错误信息），别静默忽略 |
| prompt 长度 | — | 服务端校验 `prompt_tokens ≤ max_model_len`，超长返回 400 |

### serving_chat.py — `/v1/chat/completions`

```python
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # 1. 校验不支持的参数 → 400
    # 2. chat template（在 handler 里做，纯字符串操作）：
    prompt = tokenizer.apply_chat_template(
        request.messages, tokenize=False, add_generation_prompt=True)
    # 3. 构造 SamplingParams（注意上表的映射）
    # 4. add_request → 拿 queue
    # 5. 非流式：攒完 → 组装 ChatCompletionResponse（含 usage）
    # 6. 流式：SSE，chunk 里带 delta: {"content": ...}，
    #    最后一个 chunk 带 finish_reason，然后 data: [DONE]
```

### serving_completion.py — `/v1/completions`

同上，但 prompt 不套 chat template，直接进引擎；响应是 `text` 而不是 `message.content`。

### usage 统计

- `prompt_tokens`：`add_request` 时记录（`engine.tokenizer.encode` 的长度；给 `add_request` 加个返回或让 AsyncEngine 在 tokenize 后回填）。
- `completion_tokens`：结束时 `len(completion_token_ids)`。
- 流式时客户端要 `stream_options: {"include_usage": true}` 才在最后一个 chunk 带 usage。

### 验证

用 OpenAI 官方 SDK 对拍最省事：

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

r = client.chat.completions.create(model="x", messages=[{"role": "user", "content": "你好"}],
                                   max_tokens=50)
print(r.choices[0].message.content, r.usage)

s = client.chat.completions.create(..., stream=True, max_tokens=50)
for chunk in s:
    print(chunk.choices[0].delta.content or "", end="")
```

SDK 能正常收发 = 协议兼容达标。

---

## 7. 第六步：收尾

按优先级：

1. **`GET /health`**：返回引擎线程 `is_alive()` 状态，挂了返回 5xx（方便 K8s 探针）。
2. **`GET /v1/models`**：返回单元素列表 `[{id: 模型路径, object: "model", ...}]`。
3. **abort（可选，需要动 scheduler）**：
   - handler 侧：客户端断开（`await request.is_disconnected()`）或 `DELETE /v1/completions/{request_id}` 时标记；
   - 引擎线程：从 `waiting`/`running` 移除该 seq，`block_manager.deallocate(seq)`，向队列推哨兵；
   - 给 `Scheduler` 加一个 `abort(seq_id)` 方法即可，注意先处理"正被调度中"的边界。
4. **（可选）`/metrics`**：接 prometheus-client，暴露 running/waiting 队列长度、每步 prefill/decode 吞吐（数据在 `_engine_loop` 里顺手统计）。
5. **README**：在 README.md / README_zh.md 里补服务启动用法示例。

---

## 8. 常见坑清单

| 坑 | 说明 |
|---|---|
| `step()` 空调度崩溃 | `scheduler.schedule()` 末尾 `assert scheduled_seqs`：waiting/running 都空时调用 `step()` 会 AssertionError。引擎线程必须先判 `is_finished()`，空闲时用 `threading.Event` 阻塞等待，不要轮询空转 |
| 跨线程碰 scheduler | `add_request`/`step` 只能在引擎线程调；HTTP handler 一律走入口队列。scheduler 无锁 |
| 直接在 async handler 里调 `step()` | 阻塞 CUDA 调用会卡死整个事件循环，所有请求超时 |
| `--reload` / 多 worker | 引擎和 TP 子进程被复制多份，显存翻倍或行为错乱。永远单 worker |
| 引擎在 import 时创建 | 同上，必须放 lifespan |
| `temperature=0` | `SamplingParams` 断言失败，服务端映射成 `1e-6` |
| `max_tokens` 默认 64 | 不显式传就只有 64 个 token，客户端会以为被截断 |
| decode 增量 token | 直接 decode 单 token 会把多 byte UTF-8 字符切碎，用"全量 decode + diff" |
| `--model` 传 HF hub id | `Config` 断言 `os.path.isdir`，只接受本地路径 |
| TP 子进程残留 | 显式在 shutdown 调 `engine.exit()`（`atexit` 兜底，但 lifespan shutdown 更干净、可控） |

---

## 9. 验收清单

- [ ] 两个并发请求能交错输出（continuous batching 生效，非串行）
- [ ] 长请求进行中插入短请求，短请求先完成且不影响长请求
- [ ] `stream=true` 与 `stream=false` 输出一致
- [ ] OpenAI SDK（`openai` 包）chat + completions、流式 + 非流式全部跑通
- [ ] `finish_reason` 正确：正常结束 `stop`，到 `max_tokens` 截断 `length`
- [ ] usage 数字正确（prompt/completion/total）
- [ ] 中文流式输出无乱码
- [ ] Ctrl-C / SIGTERM 后 TP 子进程全部退出（`ps` 确认无残留）
- [ ] 客户端提前断开时服务端不崩（abort 没做之前，至少日志可查、线程存活）
