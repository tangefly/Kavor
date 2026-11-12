"""Sequence 单元测试:块切分、token 推进、多模态判定、pickle 序列化往返。"""

import pickle

from kavor.engine.multimodal import MMInputs, ModalityData
from kavor.engine.sequence import SequenceStatus


def test_block_math(make_seq):
    seq = make_seq(list(range(10)))             # block_size=4
    assert seq.num_blocks == 3
    assert seq.block(0) == [0, 1, 2, 3]
    assert seq.block(1) == [4, 5, 6, 7]
    assert seq.block(2) == [8, 9]
    assert seq.last_block_num_tokens == 2
    assert len(seq) == 10


def test_append_token_updates_state(make_seq):
    seq = make_seq([1, 2, 3])
    seq.append_token(7)
    assert seq.num_tokens == 4
    assert seq.last_token == 7
    assert seq.num_completion_tokens == 1
    assert seq.prompt_token_ids == [1, 2, 3]
    assert seq.completion_token_ids == [7]


def test_is_multimodal_variants(make_seq):
    assert not make_seq([1, 2, 3]).is_multimodal
    assert make_seq([1], mm_inputs=MMInputs(image=ModalityData())).is_multimodal
    assert make_seq([1], mm_inputs=MMInputs(video=ModalityData())).is_multimodal
    # 类型 id 有非零项才算多模态
    assert make_seq([1], mm_inputs=MMInputs(mm_token_type_ids=[0, 1])).is_multimodal
    assert not make_seq([1], mm_inputs=MMInputs(mm_token_type_ids=[0, 0])).is_multimodal


def test_block_media_hashes(make_seq):
    hashes = [b"a", None, b"b", None]
    seq = make_seq([1, 2, 3, 4], mm_inputs=MMInputs(token_media_hashes=hashes))
    assert seq.block_media_hashes(0) == (b"a", None, b"b", None)
    # 全空 → None;无多模态载荷 → None
    seq = make_seq([1, 2, 3, 4], mm_inputs=MMInputs(token_media_hashes=[None] * 4))
    assert seq.block_media_hashes(0) is None
    assert make_seq([1, 2, 3, 4]).block_media_hashes(0) is None


def test_pickle_prefill_carries_full_tokens(make_seq):
    """prefill 阶段发往 TP worker:全量 token_ids + 块表,不带统计。"""
    seq = make_seq([1, 2, 3, 4, 5])
    assert seq.is_prefill
    seq.block_table = [7, 8]
    seq.state_block_ids = [9]
    seq.state_cache_ids = [10, 11]
    seq.pending_tail_copy = (1, 2)
    seq.num_cached_tokens = 4
    seq.num_scheduled_tokens = 1

    data = pickle.loads(pickle.dumps(seq))
    assert data.token_ids == [1, 2, 3, 4, 5]    # prefill 带全量
    assert data.last_token == 5
    assert data.num_tokens == 5 and data.num_prompt_tokens == 5
    assert data.num_cached_tokens == 4 and data.num_scheduled_tokens == 1
    assert data.block_table == [7, 8]
    assert data.state_block_ids == [9]
    assert data.state_cache_ids == [10, 11]
    assert data.pending_tail_copy == (1, 2)
    assert data.stats is None                   # 统计只在引擎侧


def test_pickle_decode_carries_only_last_token(make_seq):
    """decode 阶段发往 TP worker:只带 last_token,不带历史序列。"""
    seq = make_seq([1, 2, 3])
    seq.is_prefill = False
    seq.append_token(9)

    data = pickle.loads(pickle.dumps(seq))
    assert data.token_ids == []
    assert data.last_token == 9
    assert data.num_prompt_tokens == 3
    assert data.stats is None


def test_status_lifecycle(make_seq):
    seq = make_seq([1])
    assert seq.status == SequenceStatus.WAITING
    assert not seq.is_finished
    seq.status = SequenceStatus.FINISHED
    assert seq.is_finished
