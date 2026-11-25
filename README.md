# Kavor

**English** | [简体中文](README_zh.md)

Kavor is a lightweight inference framework: the entire LLM serving path — model loading, KV cache management, continuous batching, sampling, and an OpenAI-compatible server — is implemented from scratch, with no dependency on an existing inference engine. The codebase stays small enough to read end to end (~11k lines of Python, nearly a third of which is per-model network definitions). Use it as a library inside your own program, or launch a service with a single command.

## Key Features

* 🚀 **Inference performance** - Throughput comparable to vLLM
* 📖 **Code as documentation** - Every layer from attention kernels to the scheduler is hand-written, with no black-box dependencies
* ⚡ **Optimization suite** - Prefix caching, Tensor Parallelism, CUDA Graph, Torch compile, and FP8 quantization with automatic kernel selection (FlashInfer / Marlin / Triton)
* 🖼️ **Multimodal** - Image and video inputs, with the vision tower output cached across requests
* 🔌 **OpenAI compatible** - `kavor serve` exposes `/v1/chat/completions` and `/v1/models`

## Requirements

- Linux x86_64
- Python 3.10 – 3.12
- NVIDIA GPU with CUDA 12.6 or 13.0
- PyTorch 2.9.x (matched automatically during installation)

## Installation

Install directly:

```bash
pip install git+https://github.com/tangefly/Kavor.git
```

Or from source (recommended if you plan to modify anything):

```bash
git clone https://github.com/tangefly/Kavor.git
cd Kavor
pip install -e .
```

The install script detects your local `nvcc` version and picks a matching PyTorch build plus prebuilt `flash-attn` wheel, so versions line up without manual work. Override it with environment variables when needed:

| Variable | Value | Description |
| --- | --- | --- |
| `KAVOR_TORCH_CUDA` | `auto` (default) / `cu126` / `cu130` | Force a specific PyTorch CUDA build |
| `KAVOR_FLASH_ATTN` | `auto` (default) / `wheel` / `build` / `skip` | How to obtain flash-attn: prebuilt wheel, build from source, or skip |

Test and benchmark dependencies are optional:

```bash
pip install -e ".[test,bench]"
```

## Model Download

Kavor only accepts a local model directory, so download the weights first. Either of the following works.

Hugging Face:

```bash
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen3-0.6B --local-dir ~/models/Qwen3-0.6B
```

> `hf` is the CLI shipped with huggingface_hub 0.34+. On older versions use `huggingface-cli download` with the same flags.

ModelScope (usually faster inside mainland China):

```bash
pip install -U modelscope
modelscope download --model Qwen/Qwen3-0.6B --local_dir ~/models/Qwen3-0.6B
```

> Repository IDs on ModelScope mostly match Hugging Face (e.g. `Qwen/Qwen3-0.6B`), but a few third-party models use a different organization name — search on the ModelScope site if a download fails.

## Quick Start

### Serve an OpenAI-compatible API

```bash
CUDA_VISIBLE_DEVICES=0,1 kavor serve ~/models/Qwen3-0.6B \
  --served-model-name Qwen3-0.6B \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --port 8000
```

Once the server is up, send requests with the OpenAI protocol:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello, Kavor."}],
    "max_tokens": 128
  }'
```

Common flags:

| Flag | Default | Description |
| --- | --- | --- |
| `--served-model-name` | model path | Name exposed by `/v1/models`; the `model` field in requests must match it |
| `--tensor-parallel-size` / `--tp` | 1 | Tensor parallelism degree |
| `--max-model-len` | 4096 | Max context length; clamped to the model's own `max_position_embeddings` |
| `--max-num-seqs` | 512 | Max sequences batched per step |
| `--gpu-memory-utilization` | 0.9 | Fraction of GPU memory available for KV cache |
| `--enforce-eager` | off | Disable CUDA Graph and run in eager mode |
| `--temperature` | 0.0 | Server-side default sampling temperature; 0 means greedy |

### Call it from Python

```python
from kavor import LLM, SamplingParams

llm = LLM("/path/to/Qwen3-0.6B", tensor_parallel_size=1)
# temperature=0 means argmax (fully deterministic); >0 samples via Gumbel-max
sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
outputs = llm.generate(["Hello, Kavor."], sampling_params)
print(outputs[0]["text"])
```

The API mirrors vLLM's interface, with minor differences in how `LLM.generate` is called. More examples live in `examples/` (text-only, multimodal, tool calling).

## Supported Models

| Family | Hugging Face |
| --- | --- |
| Qwen3 | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) |
| Qwen3-VL | Qwen3-VL family (image / video input) |
| Qwen3.5 / Qwen3.6 | Dense and MoE variants; MoE 35B-A3B in bf16 and FP8 blockwise, with TP, CUDA Graph and image / video input |
| Mistral | [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) |
| ERNIE 4.5 | [ERNIE-4.5-0.3B-PT](https://huggingface.co/baidu/ERNIE-4.5-0.3B-PT) |
| Ouro | [Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B), [Ouro-1.4B-Thinking](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking), [Ouro-2.6B](https://huggingface.co/ByteDance/Ouro-2.6B), [Ouro-2.6B-Thinking](https://huggingface.co/ByteDance/Ouro-2.6B-Thinking) |
