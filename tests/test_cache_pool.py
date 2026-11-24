"""CachePool 的两本账:容量账本(声称的并发上限必须等于 BlockManager 真正能准入的条数)
和线性状态交接的 copy 账本。全部纯 CPU,不分配页池、不碰显存。
"""

import pytest
import torch

from kavor.engine.block_manager import BlockManager
from kavor.engine.cache_pool import CacheLayout, CachePool

NUM_POOLS = 16          # Qwen3.5-27B-FP8:16 个 full-attention 层各一个 pool


def _layout(num_blocks: int, num_state_groups: int, has_linear: bool = True) -> CacheLayout:
    return CacheLayout(
        attn_slots=[], linear_modules=[object()] if has_linear else [],
        num_pools=NUM_POOLS, num_blocks=num_blocks, block_size=1024, page_bytes=4 << 20,
        num_kv_heads=4, head_dim=256, dtype=torch.bfloat16, dtype_size=2,
        num_state_groups=num_state_groups, max_state_bytes=3 << 20,
    )


def _pool(layout: CacheLayout) -> CachePool:
    pool = CachePool(model=object(), config=object(), world_size=1)
    pool.layout = layout
    pool.linear_state_len = [-1] * layout.num_blocks
    return pool


def test_max_concurrent_matches_block_manager_admission(make_seq):
    """617 块 / (1 KV + 3 状态) = 154:既是被声称的并发,也是真正能准入的条数。"""
    num_blocks, groups = 617, 3
    capacity = _pool(_layout(num_blocks, groups)).max_concurrent()
    assert capacity == 154

    bm = BlockManager(num_blocks=num_blocks, block_size=1024,
                      has_linear_attention=True, num_state_groups=groups)
    admitted = 0
    while True:
        seq = make_seq([1])
        cached, state_ids = bm.can_allocate_prefill(seq)
        if cached == -1:
            break
        bm.allocate_prefill(seq, cached, state_ids)
        admitted += 1
        assert admitted <= capacity, "准入条数超过了 max_concurrent 声称的上限"
    assert admitted == capacity


def test_max_concurrent_without_linear_attention():
    assert _pool(_layout(8, 0, has_linear=False)).max_concurrent() == 8


# ---------------------------------------------------------------- state copy

def _record_copies(pool, monkeypatch):
    """把整页拷贝换成记录参数:这里只验证发了几次、发给谁。"""
    copies = []
    monkeypatch.setattr(pool, "copy_blocks", lambda src, dst: copies.append((list(src), list(dst))))
    return copies


def test_copy_linear_states_copies_snapshot_into_the_working_blocks(make_seq, monkeypatch):
    """冻结的快照页 -> 自己的可变块,并让新块接过旧块的状态账本。"""
    pool = _pool(_layout(32, 2))
    copies = _record_copies(pool, monkeypatch)
    plain = make_seq([1, 2, 3])
    plain.state_block_ids = [5, 6]
    resumed = make_seq(list(range(4, 12)))
    resumed.state_block_ids = [7, 8]              # 刚换出来的新块,此时还是空的
    resumed.state_copy_src = [20, 21]             # 命中的快照
    pool.linear_state_len[20:22] = [1024, 1024]

    pool.copy_linear_states([plain, resumed])
    assert copies == [([20, 21], [7, 8])]         # 没有 pending copy 的序列不参与
    assert pool.linear_state_len[7:9] == [1024, 1024]
    assert pool.linear_state_len[5:7] == [-1, -1]


def test_copy_linear_states_batches_every_pending_sequence(make_seq, monkeypatch):
    """同一步里所有待交接的序列并成一次拷贝。"""
    pool = _pool(_layout(64, 2))
    copies = _record_copies(pool, monkeypatch)
    seqs = []
    for i in range(3):
        seq = make_seq([i])
        seq.state_block_ids = [2 * i, 2 * i + 1]
        seq.state_copy_src = [30 + 2 * i, 31 + 2 * i]
        pool.linear_state_len[30 + 2 * i:32 + 2 * i] = [512 * (i + 1)] * 2
        seqs.append(seq)

    pool.copy_linear_states(seqs)
    assert copies == [([30, 31, 32, 33, 34, 35], [0, 1, 2, 3, 4, 5])]
    assert pool.linear_state_len[:6] == [512, 512, 1024, 1024, 1536, 1536]


def test_copy_linear_states_without_pending_copies_issues_nothing(make_seq, monkeypatch):
    pool = _pool(_layout(32, 2))
    monkeypatch.setattr(pool, "copy_blocks",
                        lambda src, dst: pytest.fail("没有待交接的状态,不该发拷贝"))
    seq = make_seq([1, 2, 3])
    seq.state_block_ids = [7, 8]
    pool.copy_linear_states([seq])

    # full attention 模型没有状态块, 同样是空操作
    plain = _pool(_layout(8, 0, has_linear=False))
    monkeypatch.setattr(plain, "copy_blocks",
                        lambda src, dst: pytest.fail("full attention 模型不该发拷贝"))
    plain.copy_linear_states([make_seq([1])])
