"""GPU 端到端测试:需要真实模型权重,用环境变量门控。

用法(权重路径每台机器不同,通过环境变量传入):
    KAVOR_TEST_MODEL=~/huggingface/Qwen3-0.6B pytest tests/test_e2e_gpu.py -v
未设置该变量时自动跳过,不影响 CPU 单元测试。
"""

import os

import pytest

MODEL_PATH = os.environ.get("KAVOR_TEST_MODEL")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH,
    reason="未设置 KAVOR_TEST_MODEL(模型权重路径),跳过 GPU 端到端测试",
)


@pytest.fixture(scope="module")
def llm():
    from kavor import LLM

    engine = LLM(MODEL_PATH, enforce_eager=True, tensor_parallel_size=1)
    yield engine
    engine.exit()


def test_basic_generate(llm):
    from kavor import SamplingParams

    outputs = llm.generate(
        ["Hello, Kavor.", "Kavor is a lightweight vLLM implementation."],
        SamplingParams(temperature=0.6, max_tokens=16),
        use_tqdm=False,
    )
    assert len(outputs) == 2
    for output in outputs:
        assert isinstance(output["text"], str)
        # 中途命中 eos 可能提前结束,但至少出一个 token
        assert 1 <= len(output["token_ids"]) <= 16


def test_batch_generate(llm):
    from kavor import SamplingParams

    prompts = [f"Prompt number {i}: the quick brown fox." for i in range(4)]
    outputs = llm.generate(prompts, SamplingParams(temperature=0.6, max_tokens=8),
                           use_tqdm=False)
    assert len(outputs) == 4
    assert all(1 <= len(output["token_ids"]) <= 8 for output in outputs)


def test_prefix_cache_hit_on_repeated_prompt_full_attention(llm):
    from kavor import SamplingParams

    if llm.config.has_linear_attention:
        pytest.skip("线性注意力模型的前缀复用走 bundle 路径,由下一个测试覆盖")

    # 默认 block_size=256,prompt 要超过一个块才有整块前缀可命中
    prompt = "Prefix caching test. Kavor reuses KV blocks across requests. " * 40
    params = SamplingParams(temperature=0.6, max_tokens=8)

    llm.generate([prompt], params, use_tqdm=False)
    hits_before = llm.scheduler.block_manager.prefix_cache_hit_tokens

    llm.generate([prompt], params, use_tqdm=False)
    hits_after = llm.scheduler.block_manager.prefix_cache_hit_tokens

    assert hits_after > hits_before


def test_prefix_cache_hit_on_repeated_prompt_linear(llm):
    """线性注意力模型:前缀复用依赖快照边界 token(<think>, id=248068)触发的
    bundle 状态快照——clamp_tokens 会把第一段 prefill 停在它之前并记录
    linear states,第二个相同 prompt 命中 bundle,partial token 免算。"""
    from kavor import SamplingParams
    from kavor.engine.multimodal import PromptInputs

    if not llm.config.has_linear_attention:
        pytest.skip("full attention 模型的前缀复用走 KV 块路径,由上一个测试覆盖")

    snap_token = 248068   # scheduler.clamp_tokens 的快照边界 token(Qwen3.5 <think>)
    tokenizer = llm.tokenizer
    lead = tokenizer.encode("Linear attention prefix caching test. " * 60)
    assert len(lead) >= 300
    tail = tokenizer.encode(" Kavor reuses linear states across requests.")
    # 快照点放在 300:跨过一个整块(256)再留 44 个 partial token,
    # 同时覆盖 KV 块复用与 bundle partial 匹配两段逻辑
    token_ids = lead[:300] + [snap_token] + tail

    inputs = PromptInputs(token_ids=token_ids, prompt="")
    params = SamplingParams(temperature=0.6, max_tokens=8)

    llm.generate([inputs], params, use_tqdm=False)
    hits_before = llm.scheduler.block_manager.prefix_cache_hit_tokens

    llm.generate([inputs], params, use_tqdm=False)
    hits_after = llm.scheduler.block_manager.prefix_cache_hit_tokens

    assert hits_after > hits_before
