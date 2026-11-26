# Kavor

[English](README.md) | **简体中文**

Kavor 是一个轻量级推理框架：从零实现大模型推理的完整链路——模型加载、KV cache 管理、连续批处理调度、采样、以及 OpenAI 兼容服务，全部手写，不依赖 vLLM 之类的推理引擎。代码量保持在可通读的规模内（约 1.1 万行 Python，其中近三成是各模型的网络结构实现）。既能作为库嵌入你自己的程序，也能一条命令拉起服务对外提供 API。

## 主要特性

* 🚀 **推理性能** - 吞吐与 vLLM 相当
* 📖 **代码即文档** - 从 attention kernel 到调度器的每一层都是手写实现，无黑盒依赖，适合精读与二次开发
* ⚡ **优化全家桶** - 前缀缓存（Prefix Caching）、张量并行（Tensor Parallelism）、CUDA Graph、Torch compile、FP8 量化（FlashInfer / Marlin / Triton 多后端自动择优）
* 🖼️ **多模态** - 支持图片与视频输入，视觉塔输出可跨请求缓存复用
* 🔌 **OpenAI 兼容** - `kavor serve` 直接提供 `/v1/chat/completions`、`/v1/models`

## 环境要求

- Linux x86_64（aarch64 也可用，但 flash-attn 需要现场编译）
- Python 3.10 – 3.13
- NVIDIA GPU（compute capability 8.0 及以上）
- PyTorch 2.14（安装时自动选择，见下文）

## 安装

直接安装：

```bash
pip install git+https://github.com/tangefly/Kavor.git
```

从源码安装（推荐，便于二次开发）：

```bash
git clone https://github.com/tangefly/Kavor.git
cd Kavor
pip install -e .
```

一条命令装齐全部依赖。构建阶段脚本会：

1. 探测目标环境的 `torch`，并读取本机 `nvcc` 与驱动版本；
2. 把 `torch` 固定为 2.14.1，并选对应的 CUDA 构建（`cu126` / `cu130`）——`nvcc` 优先，驱动版本作为硬上限；
3. 安装 `flash-attn`：[Kavor release](https://github.com/tangefly/Kavor/releases) 上有可用的 wheel 就直接用；没有就在同一次执行里拉取 `v2.8.3.post2` 版本的 `flash-attention` 仓库现场编译（10–30 分钟，产物缓存在 `~/.cache/kavor/wheels`）。

全新环境下脚本会先自己装好 `torch` 再编译，所以一条 `pip install -e .` 就够。

```bash
KAVOR_TORCH_CUDA=cu126 pip install -e .   # 强制指定 CUDA 构建(cu126 / cu130 / none)
KAVOR_INSTALL_FLASH_ATTN=0 pip install -e .   # 不装 flash-attn(pip download、uv lock)
```

| 变量 | 取值 | 说明 |
| --- | --- | --- |
| `KAVOR_TORCH_CUDA` | `auto`（默认）/ `cu126` / `cu130` / `none` | 强制或跳过程序对 PyTorch CUDA 构建的选择 |
| `KAVOR_INSTALL_FLASH_ATTN` | `1`（默认）/ `0` | 是否安装 `flash-attn` |
| `KAVOR_WHEEL_CACHE` | 路径 | 现场编译产物的缓存目录（默认 `~/.cache/kavor/wheels`） |

测试与压测依赖按需安装：

```bash
pip install -e ".[test,bench]"
```

## 模型下载

Kavor 只接受本地模型目录，权重需要先下载。以下两种方式任选其一。

Hugging Face：

```bash
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen3-0.6B --local-dir ~/models/Qwen3-0.6B
```

> `hf` 是 huggingface_hub 0.34+ 的新命令；更早的版本请用 `huggingface-cli download`，参数相同。

ModelScope（国内网络更快）：

```bash
pip install -U modelscope
modelscope download --model Qwen/Qwen3-0.6B --local_dir ~/models/Qwen3-0.6B
```

> ModelScope 上的仓库 ID 大多与 Hugging Face 一致（例如 `Qwen/Qwen3-0.6B`），少数第三方模型的组织名可能不同，以 ModelScope 站点的搜索结果为准。

## 启动方法

### 启动 OpenAI 兼容服务

```bash
CUDA_VISIBLE_DEVICES=0,1 kavor serve ~/models/Qwen3-0.6B \
  --served-model-name Qwen3-0.6B \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --port 8000
```

服务就绪后即可按 OpenAI 协议调用：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello, Kavor."}],
    "max_tokens": 128
  }'
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--served-model-name` | 模型路径 | `/v1/models` 暴露的名字，请求体里的 `model` 需与之一致 |
| `--tensor-parallel-size` / `--tp` | 1 | 张量并行度 |
| `--max-model-len` | 4096 | 最大上下文长度；超过模型自身的 `max_position_embeddings` 时会被截断到后者 |
| `--max-num-seqs` | 512 | 单步最大并发序列数 |
| `--gpu-memory-utilization` | 0.9 | 可用于 KV cache 的显存比例 |
| `--enforce-eager` | 关 | 关闭 CUDA Graph，全部走 eager 执行 |
| `--temperature` | 0.0 | 服务端默认采样温度，0 表示 greedy |

### Python API 调用

```python
from kavor import LLM, SamplingParams

llm = LLM("/path/to/Qwen3-0.6B", tensor_parallel_size=1)
# temperature=0 走 argmax，输出完全确定；>0 时按 Gumbel-max 采样
sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
outputs = llm.generate(["Hello, Kavor."], sampling_params)
print(outputs[0]["text"])
```

接口与 vLLM 保持一致，仅 `LLM.generate` 的调用形式略有差异。更多示例见 `examples/`（纯文本、多模态、工具调用）。

## 支持的模型

| 家族 | Hugging Face |
| --- | --- |
| Qwen3 | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) |
| Qwen3-VL | Qwen3-VL 系列（图片 / 视频输入） |
| Qwen3.5 / Qwen3.6 | 稠密与 MoE 版本；MoE 35B-A3B 支持 bf16 与 FP8 blockwise，TP、CUDA Graph、图文 / 视频输入 |
| Mistral | [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) |
| ERNIE 4.5 | [ERNIE-4.5-0.3B-PT](https://huggingface.co/baidu/ERNIE-4.5-0.3B-PT) |
| Ouro | [Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B)、[Ouro-1.4B-Thinking](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking)、[Ouro-2.6B](https://huggingface.co/ByteDance/Ouro-2.6B)、[Ouro-2.6B-Thinking](https://huggingface.co/ByteDance/Ouro-2.6B-Thinking) |
