"""GPU 端到端测试:需要真实模型权重,用环境变量门控。

用法(权重路径每台机器不同,通过环境变量传入):
    KAVOR_TEST_MODEL=~/huggingface/Qwen3-0.6B pytest tests/test_e2e_gpu.py -v
未设置该变量时自动跳过,不影响 CPU 单元测试。

线性注意力模型的用例默认以 eager 模式跑;KAVOR_TEST_EAGER=0 时开着 CUDA graph 把同一套
用例再跑一遍(decode 跨 chunk 边界后的状态拷贝发生在 graph replay 之前,两条路径都要过)。
KAVOR_TEST_TP=2 时用多卡 tensor parallel 跑,此时默认开 CUDA graph(Sequence 发往 TP worker 的
pickle 字段也在这些用例的覆盖范围内)。
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

    tp = int(os.environ.get("KAVOR_TEST_TP", "1"))
    # eager + TP>1 现在跑不通:非 0 号 rank 没有 logits,ModelRunner.run 却会去读 graph_vars,
    # worker 直接崩、0 号 rank 卡在通信里。多卡默认开 CUDA graph,除非显式要求 eager。
    default_eager = "0" if tp > 1 else "1"
    engine = LLM(MODEL_PATH, enforce_eager=os.environ.get("KAVOR_TEST_EAGER", default_eager) != "0",
                 tensor_parallel_size=tp)
    yield engine
    engine.exit()


@pytest.fixture(autouse=True)
def real_block_size(small_block_size, llm, monkeypatch):
    """把 Sequence.block_size 改回 engine 实际使用的值。

    conftest 的 small_block_size 为了 CPU 单测把 Sequence.block_size 改成 4,对同目录的
    GPU 用例同样生效;但这里的 engine 是在它之前按真实的 block_size 建的。两边不一致时
    块表和 decode 的 KV 槽位都会算错:eager 下不报错,只是生成的 token 的 KV 写丢了,
    CUDA graph 下块表拷不进图里的缓冲区直接崩。显式依赖 small_block_size 保证改在它之后。
    """
    from kavor.engine.sequence import Sequence

    monkeypatch.setattr(Sequence, "block_size", llm.config.kvcache_block_size)


def greedy(max_tokens: int):
    """逐 token 对比需要确定性解码:同一个输入每次必须给出同一个 token。

    temperature == 0 走 argmax(平局取最小下标),输出与输入严格一一对应。
    """
    from kavor import SamplingParams

    return SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)


def linear_only(llm):
    if not llm.config.has_linear_attention:
        pytest.skip("full attention 模型的前缀复用走 KV 块路径,由别的用例覆盖")


def chunk_aligned_prompt(llm, num_chunks: int = 2, tail_tokens: int = 64,
                         phrase: str = "Linear attention prefix caching test. "):
    """一段文本 prompt:跨过 num_chunks 个 chunk,再多 tail_tokens 个 token。

    前者可复用;尾巴不满一个 chunk,按设计不参与复用。同一个 engine 里前面的用例留下
    的缓存会留给后面的用例,需要从头算的用例自己换一个 phrase。
    """
    from kavor.engine.multimodal import PromptInputs

    tokenizer = llm.tokenizer
    phrase_ids = tokenizer.encode(phrase)
    need = num_chunks * llm.config.kvcache_block_size + tail_tokens
    # 按 token 重复:重复的字符串直接编码会在接缝处合并 token,凑不够想要的长度
    token_ids = phrase_ids * -(-need // len(phrase_ids))
    token_ids = token_ids[:need] + tokenizer.encode(" Kavor reuses linear states across requests.")
    return PromptInputs(token_ids=token_ids, prompt="")


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
        pytest.skip("线性注意力模型的前缀复用走 chunk 状态快照路径,由后面的用例覆盖")

    # 默认 block_size=256,prompt 要超过一个块才有整块前缀可命中
    prompt = "Prefix caching test. Kavor reuses KV blocks across requests. " * 40
    params = SamplingParams(temperature=0.6, max_tokens=8)

    llm.generate([prompt], params, use_tqdm=False)
    hits_before = llm.scheduler.block_manager.prefix_cache_hit_tokens

    llm.generate([prompt], params, use_tqdm=False)
    hits_after = llm.scheduler.block_manager.prefix_cache_hit_tokens

    assert hits_after > hits_before


def test_prefix_cache_hit_on_repeated_prompt_linear(llm):
    """线性注意力模型:状态只在 chunk 边界(block_size 的整倍数)上留快照。

    每段 prefill 停在边界上,第二个相同 prompt 从最后一个整 chunk 的边界续算;
    尾部不满一个 chunk 的部分不复用,所以命中量正好是 num_chunks * block_size。
    """
    from kavor import SamplingParams

    linear_only(llm)
    block_size = llm.config.kvcache_block_size
    inputs = chunk_aligned_prompt(llm)
    params = SamplingParams(temperature=0.6, max_tokens=8)

    llm.generate([inputs], params, use_tqdm=False)
    hits_before = llm.scheduler.block_manager.prefix_cache_hit_tokens

    llm.generate([inputs], params, use_tqdm=False)
    hits_after = llm.scheduler.block_manager.prefix_cache_hit_tokens

    assert hits_after - hits_before == 2 * block_size


def test_linear_prefix_cache_hit_preserves_output(llm):
    """命中前缀缓存不能改变输出。

    第一遍没有任何缓存,必然从头算;第二遍命中第一遍留下的快照。两遍算最后一段时
    输入完全相同(KV 块直接复用,线性状态是同一份快照的拷贝),所以要求逐 token 相等。
    """
    linear_only(llm)
    block_size = llm.config.kvcache_block_size
    inputs = chunk_aligned_prompt(llm)
    params = greedy(max_tokens=16)

    first = llm.generate([inputs], params, use_tqdm=False)[0]["token_ids"]
    hits_before = llm.scheduler.block_manager.prefix_cache_hit_tokens
    second = llm.generate([inputs], params, use_tqdm=False)[0]["token_ids"]

    assert len(first) == 16
    assert llm.scheduler.block_manager.prefix_cache_hit_tokens - hits_before == 2 * block_size
    assert second == first


def test_linear_identical_prompts_in_one_batch_agree(llm):
    """同一个 batch 里两条完全相同的 prompt:各自从头算,输出必须一致。

    它们在同一步走到同一个 chunk 边界,状态快照只登记一次,另一条在原来的块上原地继续。
    之后再来的相同请求要能命中这个快照。
    """
    linear_only(llm)
    block_size = llm.config.kvcache_block_size
    inputs = chunk_aligned_prompt(llm, phrase="Linear attention batch prefix caching test. ")
    params = greedy(max_tokens=8)
    hits_before = llm.scheduler.block_manager.prefix_cache_hit_tokens

    outputs = llm.generate([inputs, inputs], params, use_tqdm=False)

    assert llm.scheduler.block_manager.prefix_cache_hit_tokens == hits_before   # 都从头算
    assert [len(output["token_ids"]) for output in outputs] == [8, 8]
    assert outputs[1]["token_ids"] == outputs[0]["token_ids"]

    # batch 大小不同, GEMM 的数值不完全一样, 这里只验证能从快照续算, 不比输出
    hits_before = llm.scheduler.block_manager.prefix_cache_hit_tokens
    again = llm.generate([inputs], params, use_tqdm=False)[0]["token_ids"]
    assert llm.scheduler.block_manager.prefix_cache_hit_tokens - hits_before == 2 * block_size
    assert len(again) == 8


def test_linear_next_turn_resumes_from_a_decode_boundary(llm):
    """多轮对话:上一轮的 prompt 不足一个 chunk,要到 decode 阶段才跨过第一个边界。

    下一轮带着完整历史回来,从这个边界续算,续出来的 token 要和上一轮接着生成的一样。
    没有 decode 阶段的冻结,这里一处可复用的状态都没有。
    """
    linear_only(llm)
    block_size = llm.config.kvcache_block_size
    tokenizer = llm.tokenizer
    prompts = [
        "The quick brown fox jumps over the lazy dog. ",
        "Photosynthesis lets plants turn sunlight into chemical energy. ",
        "In 1969, Apollo 11 landed the first humans on the Moon. ",
    ]
    from kavor.engine.multimodal import PromptInputs

    for text in prompts:
        prompt_ids = tokenizer.encode(text)
        assert len(prompt_ids) < block_size             # 边界在 prompt 之后才会被跨过
        num_tokens = block_size + 4 - len(prompt_ids)   # 历史长度 = block_size + 4
        first = llm.generate(
            [PromptInputs(token_ids=prompt_ids, prompt=text)],
            greedy(max_tokens=num_tokens + 1), use_tqdm=False)[0]["token_ids"]
        assert len(first) == num_tokens + 1

        # 下一轮:prompt = 上一轮的 prompt + 前 num_tokens 个生成 token
        history = PromptInputs(token_ids=prompt_ids + first[:num_tokens], prompt="")
        hits_before = llm.scheduler.block_manager.prefix_cache_hit_tokens
        second = llm.generate([history], greedy(max_tokens=1), use_tqdm=False)[0]["token_ids"]

        assert llm.scheduler.block_manager.prefix_cache_hit_tokens - hits_before == block_size
        assert second[0] == first[num_tokens]
