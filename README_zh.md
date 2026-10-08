# Kavor

[English](README.md) | **简体中文**

一个从零开始实现的轻量级 vLLM。

## 主要特性

* 🚀 **快速离线推理** - 推理速度与 vLLM 相当
* 📖 **代码可读性强** - 仅用约 1,200 行 Python 代码实现，简洁清晰
* ⚡ **优化全家桶** - 前缀缓存（Prefix Caching）、张量并行（Tensor Parallelism）、Torch 编译、CUDA Graph 等

## 安装

```bash
pip install git+https://github.com/tangefly/Kavor.git
```

## 模型下载

如需手动下载模型权重，可使用以下命令：
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## 快速上手

- API 代码调用

用法参见 `example.py`。API 与 vLLM 接口保持一致，仅在 `LLM.generate` 方法上略有差异：
```python
from kavor import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Kavor."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

- HTTP 调用

```
CUDA_VISIBLE_DEVICES=0,1 kavor serve Mistral-7B-Instruct-v0.2 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --port 25541
```

## 支持的模型

| 家族 | Hugging Face | ModelScope |
|----------------|-------------|-------------|
| Qwen3           | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)     |      [Qwen3-0.6B](https://www.modelscope.cn/models/Qwen/Qwen3-0.6B)      |
| Mistral         | [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)     |      [Mistral-7B-Instruct-v0.2](https://www.modelscope.cn/models/AI-ModelScope/Mistral-7B-Instruct-v0.2/summary)      |

## 性能测试

基准测试参见 `bench.py`。

**测试配置：**
- 硬件：RTX 4070 Laptop (8GB)
- 模型：Qwen3-0.6B
- 总请求数：256 条序列
- 输入长度：在 100–1024 tokens 之间随机采样
- 输出长度：在 100–1024 tokens 之间随机采样

**性能结果：**
| 推理引擎 | 输出 Tokens | 耗时 (s) | 吞吐量 (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Kavor          | 133,966     | 93.41    | 1434.13               |
