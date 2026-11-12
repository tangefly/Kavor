"""公共 fixture 与构造工具。

要点:
- 生产环境 block_size=256,测试统一用 4,几十个 token 就能覆盖多块逻辑。
- Sequence.block_size 是类属性(LLMEngine 初始化时按配置改写),用 autouse
  fixture 补丁成小值,并保证 BlockManager 的 block_size 与之一致。
- Config.__post_init__ 要求真实模型目录,单测用 SimpleNamespace 只提供
  Scheduler 实际读取的字段。
"""

from types import SimpleNamespace

import pytest

from kavor.engine.sequence import Sequence
from kavor.sampling_params import SamplingParams

TEST_BLOCK_SIZE = 4


@pytest.fixture(autouse=True)
def small_block_size(monkeypatch):
    monkeypatch.setattr(Sequence, "block_size", TEST_BLOCK_SIZE)


@pytest.fixture
def make_inputs():
    from kavor.engine.multimodal import PromptInputs

    def _make(token_ids, mm_inputs=None):
        return PromptInputs(token_ids=list(token_ids), prompt="", mm_inputs=mm_inputs)

    return _make


@pytest.fixture
def make_seq(make_inputs):
    def _make(token_ids, *, temperature=1.0, max_tokens=64, ignore_eos=False,
              mm_inputs=None):
        return Sequence(
            make_inputs(token_ids, mm_inputs=mm_inputs),
            SamplingParams(temperature=temperature, max_tokens=max_tokens,
                           ignore_eos=ignore_eos),
        )

    return _make


@pytest.fixture
def make_config():
    def _make(**overrides):
        values = dict(
            max_num_seqs=8,
            max_num_batched_tokens=256,
            eos=0,
            kvcache_block_size=TEST_BLOCK_SIZE,
            num_kvcache_blocks=64,
            has_linear_attention=False,
            num_linear_state_groups=0,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    return _make
