"""BlockManager 单元测试:块分配、prefix cache 命中/复用、linear attention bundle。

全部纯 CPU,不依赖模型与 GPU。
"""

import pytest

from kavor.engine.block_manager import BlockManager, NONE_HASH

TEST_BLOCK_SIZE = 4


def _prefill(bm: BlockManager, seq):
    """模拟一次完整的单段 prefill:分配 + 整块哈希(对应 postprocess 的前半段)。"""
    cached, bundle = bm.can_allocate_prefill(seq)
    assert cached != -1
    bm.allocate_prefill(seq, cached, bundle)
    seq.num_scheduled_tokens = seq.num_tokens - seq.num_cached_tokens
    bm.hash_blocks(seq)
    seq.num_cached_tokens += seq.num_scheduled_tokens
    seq.num_scheduled_tokens = 0


# ---------------------------------------------------------------- 基础分配

def test_allocate_and_deallocate_roundtrip():
    bm = BlockManager(num_blocks=8, block_size=TEST_BLOCK_SIZE)
    assert len(bm.free_block_ids) == 8
    ids = [bm.allocate_block() for _ in range(8)]
    assert sorted(ids) == list(range(8))
    assert not bm.free_block_ids
    for block_id in ids:
        # deallocate_block 只接受引用数已归零的块(高层 deallocate 负责递减)
        bm.blocks[block_id].ref_count = 0
        bm.deallocate_block(block_id)
    assert len(bm.free_block_ids) == 8


def test_linear_attention_reserves_padding_block():
    bm = BlockManager(num_blocks=8, block_size=TEST_BLOCK_SIZE,
                      has_linear_attention=True, num_state_groups=2)
    assert bm.padding_block_id == 7
    # 最后一块是占位块,永不进入 free list
    assert set(bm.free_block_ids) == set(range(7))
    for _ in range(7):
        bm.allocate_block()
    with pytest.raises(IndexError):
        bm.allocate_block()


def test_hash_block_tokens_chains_parent_tokens_and_media():
    h1 = BlockManager.hash_block_tokens(-1, [1, 2, 3, 4])
    assert h1 == BlockManager.hash_block_tokens(-1, [1, 2, 3, 4])
    # token 不同 → 哈希不同
    assert h1 != BlockManager.hash_block_tokens(-1, [1, 2, 3, 5])
    # 父块哈希参与链式计算
    assert h1 != BlockManager.hash_block_tokens(h1, [1, 2, 3, 4])
    # 多模态媒体哈希参与计算
    assert h1 != BlockManager.hash_block_tokens(-1, [1, 2, 3, 4], (b"media",))


# ---------------------------------------------------------------- prefix cache

def test_prefix_cache_reuses_finished_prefix(make_seq):
    bm = BlockManager(num_blocks=16, block_size=TEST_BLOCK_SIZE)
    seq1 = make_seq(list(range(12)))            # 3 个整块
    _prefill(bm, seq1)
    reused_ids = seq1.block_table[:2]
    bm.deallocate(seq1)
    assert seq1.block_table == []               # deallocate 清空块表

    # 相同 prompt 的第二个请求:前 num_blocks-1 块命中(最后一块不缓存)
    seq2 = make_seq(list(range(12)))
    cached, bundle = bm.can_allocate_prefill(seq2)
    assert (cached, bundle) == (2, None)
    bm.allocate_prefill(seq2, cached, bundle)
    assert seq2.block_table[:2] == reused_ids   # 物理块被复用
    assert seq2.num_cached_tokens == 8
    assert all(bm.blocks[bid].ref_count == 1 for bid in reused_ids)
    # 命中统计:两次请求累计 prompt 24,hit 8
    assert bm.prefix_cache_hit_tokens == 8
    assert bm.prefix_cache_prompt_tokens == 24


def test_prefix_cache_hits_only_up_to_divergence(make_seq):
    bm = BlockManager(num_blocks=16, block_size=TEST_BLOCK_SIZE)
    seq1 = make_seq(list(range(12)))
    _prefill(bm, seq1)
    bm.deallocate(seq1)

    seq2 = make_seq([0, 1, 2, 3, 40, 41, 42, 43, 8, 9, 10, 11])  # 第 2 块分叉
    cached, _ = bm.can_allocate_prefill(seq2)
    assert cached == 1


def test_last_block_of_longer_prompt_is_reusable(make_seq):
    """短请求的最后一个整块虽是"尾块",对更长的同前缀请求依然可命中。"""
    bm = BlockManager(num_blocks=16, block_size=TEST_BLOCK_SIZE)
    seq1 = make_seq(list(range(8)))             # 2 个整块
    _prefill(bm, seq1)
    bm.deallocate(seq1)

    seq2 = make_seq(list(range(12)))            # 前缀覆盖 seq1 全部
    cached, _ = bm.can_allocate_prefill(seq2)
    assert cached == 2


def test_hash_entry_with_mismatched_tokens_is_rejected(make_seq):
    """哈希表条目指向的块内容对不上时(极端哈希冲突),靠 token 校验拒绝。"""
    bm = BlockManager(num_blocks=16, block_size=TEST_BLOCK_SIZE)
    seq1 = make_seq(list(range(12)))
    _prefill(bm, seq1)
    block1_hash = bm.blocks[seq1.block_table[1]].hash
    # 人为把第 2 块的哈希指到内容不同的块 0
    bm.hash_to_block_id[block1_hash] = seq1.block_table[0]

    seq2 = make_seq(list(range(12)))
    cached, _ = bm.can_allocate_prefill(seq2)
    assert cached == 1                          # 块 0 命中后在校验处停下


def test_shared_prefix_with_running_seq_increases_refcount(make_seq):
    bm = BlockManager(num_blocks=4, block_size=TEST_BLOCK_SIZE)
    seq1 = make_seq(list(range(12)))
    _prefill(bm, seq1)                          # 占 3 块,剩 1 空闲

    seq2 = make_seq(list(range(12)))
    cached, _ = bm.can_allocate_prefill(seq2)
    assert cached == 2
    bm.allocate_prefill(seq2, cached, None)     # 共享 2 块 + 新分配 1 块
    assert not bm.free_block_ids
    shared = seq1.block_table[:2]
    assert all(bm.blocks[bid].ref_count == 2 for bid in shared)
    assert all(bid in bm.used_block_ids for bid in shared)

    bm.deallocate(seq1)                         # 引用减一,块仍被 seq2 持有
    assert all(bm.blocks[bid].ref_count == 1 for bid in shared)
    assert all(bid in bm.used_block_ids for bid in shared)
    bm.deallocate(seq2)
    assert len(bm.free_block_ids) == 4


def test_can_allocate_prefill_reports_shortage(make_seq):
    bm = BlockManager(num_blocks=2, block_size=TEST_BLOCK_SIZE)
    assert bm.can_allocate_prefill(make_seq(list(range(12)))) == (-1, None)


def test_allocate_block_evicts_stale_hash_entry(make_seq):
    """空闲块被重新分配时,要把它残留的哈希条目从表中摘掉。"""
    bm = BlockManager(num_blocks=16, block_size=TEST_BLOCK_SIZE)
    seq1 = make_seq(list(range(12)))
    _prefill(bm, seq1)
    block_id, block_hash = seq1.block_table[0], bm.blocks[seq1.block_table[0]].hash
    bm.deallocate(seq1)                         # 块空闲但哈希保留,可供前缀复用
    assert bm.hash_to_block_id[block_hash] == block_id

    bm.free_block_ids.remove(block_id)
    bm.free_block_ids.appendleft(block_id)      # 让它下一个被分配
    assert bm.allocate_block() == block_id
    assert bm.blocks[block_id].hash == -1
    assert block_hash not in bm.hash_to_block_id


def test_may_append_block_grows_only_at_boundary(make_seq):
    bm = BlockManager(num_blocks=8, block_size=TEST_BLOCK_SIZE)
    seq = make_seq(list(range(4)))
    bm.allocate_prefill(seq, 0, None)
    seq.num_cached_tokens = 4
    seq.append_token(50)                        # 5 个 token,新 token 跨块

    assert len(seq.block_table) == 1
    assert bm.can_append_block(seq)
    bm.may_append_block(seq)
    assert len(seq.block_table) == 2

    seq.append_token(51)                        # 6 个 token,不跨块
    bm.may_append_block(seq)
    assert len(seq.block_table) == 2


# ---------------------------------------------------------------- linear attention bundle

SNAP_TOKEN = 248068   # 与 scheduler.clamp_tokens 中 Qwen3.5 的快照边界 token 一致


def _record_start_bundle(bm, seq, num_chunk_tokens):
    """模拟分块 prefill 的第一段:停在快照边界前,记录 linear states bundle。"""
    seq.num_scheduled_tokens = num_chunk_tokens
    bm.hash_blocks(seq)                         # 不足整块时不产生哈希
    seq.num_cached_tokens += num_chunk_tokens
    bm.record_linear_cached_blocks(seq)
    seq.num_scheduled_tokens = 0


def test_linear_prefill_needs_state_block_budget(make_seq):
    # 1 个 KV 块 + 2 组 state × (推理 + 快照) = 5 块;池 6 减占位块刚好剩 5
    bm = BlockManager(num_blocks=6, block_size=TEST_BLOCK_SIZE,
                      has_linear_attention=True, num_state_groups=2)
    seq = make_seq([1, 2, 3])
    cached, bundle = bm.can_allocate_prefill(seq)
    assert (cached, bundle) == (0, None)
    bm.allocate_prefill(seq, 0, None)
    assert len(seq.state_block_ids) == 2
    assert len(seq.state_cache_ids) == 2

    bm2 = BlockManager(num_blocks=5, block_size=TEST_BLOCK_SIZE,
                       has_linear_attention=True, num_state_groups=2)
    assert bm2.can_allocate_prefill(make_seq([1, 2, 3])) == (-1, None)


def test_linear_bundle_record_search_and_adopt(make_seq):
    bm = BlockManager(num_blocks=24, block_size=TEST_BLOCK_SIZE,
                      has_linear_attention=True, num_state_groups=2)
    tokens1 = [1, 2, 3, SNAP_TOKEN, 5, 6, 7, 8, 9]
    seq1 = make_seq(tokens1)
    bm.allocate_prefill(seq1, 0, None)
    tail_id = seq1.block_table[0]
    donor_state_cache_ids = list(seq1.state_cache_ids)

    # 第一段 prefill 停在快照边界前,bundle 记录 parent=-1、partial=[1,2,3]
    _record_start_bundle(bm, seq1, 3)
    (bundle,) = bm.linear_hash_table[-1]
    assert bundle.partial_token_ids == [1, 2, 3]
    assert bundle.tail_block_id == tail_id
    assert bm.blocks[tail_id].bundle is bundle
    assert bm.search_linear_states(-1, make_seq(tokens1), 0) is bundle

    # 第二段跑完整个 prefill 并结束,块全部释放但 bundle 惰性保留
    seq1.num_scheduled_tokens = len(tokens1) - seq1.num_cached_tokens
    bm.hash_blocks(seq1)
    seq1.num_cached_tokens += seq1.num_scheduled_tokens
    bm.deallocate(seq1)
    assert seq1.state_block_ids is None and seq1.state_cache_ids is None
    assert bm.search_linear_states(-1, make_seq(tokens1), 0) is bundle

    # 新请求共享 [1,2,3] 前缀:在第一块之前就命中 bundle
    seq2 = make_seq(tokens1 + [10, 11])
    cached, bundle = bm.can_allocate_prefill(seq2)
    assert cached == 0 and bundle is not None
    bm.allocate_prefill(seq2, cached, bundle)

    # 命中即偷走:bundle 从表中摘除,新 seq 继承 state 块与部分尾块
    assert bm.search_linear_states(-1, make_seq(tokens1), 0) is None
    assert seq2.num_cached_tokens == 3          # partial 3 个 token 免算
    assert seq2.pending_tail_copy is not None
    assert seq2.pending_tail_copy[0] == tail_id
    assert seq2.state_block_ids == donor_state_cache_ids
    assert len(seq2.state_cache_ids) == 2

    bm.deallocate(seq2)
    assert len(bm.free_block_ids) == 23         # 全部归还(占位块除外)


def test_bundle_covering_whole_prompt_is_ignored(make_seq):
    bm = BlockManager(num_blocks=24, block_size=TEST_BLOCK_SIZE,
                      has_linear_attention=True, num_state_groups=2)
    seq1 = make_seq([1, 2, 3, SNAP_TOKEN, 5, 6, 7, 8, 9])
    bm.allocate_prefill(seq1, 0, None)
    _record_start_bundle(bm, seq1, 3)

    # 新请求恰好只覆盖 partial:无需计算,bundle 被忽略,从头 prefill
    seq2 = make_seq([1, 2, 3])
    cached, bundle = bm.can_allocate_prefill(seq2)
    assert cached == 0 and bundle is None


def test_bundle_lazily_evicted_when_blocks_reused(make_seq):
    bm = BlockManager(num_blocks=24, block_size=TEST_BLOCK_SIZE,
                      has_linear_attention=True, num_state_groups=2)
    seq1 = make_seq([1, 2, 3, SNAP_TOKEN, 5, 6, 7, 8, 9])
    bm.allocate_prefill(seq1, 0, None)
    _record_start_bundle(bm, seq1, 3)
    assert bm.search_linear_states(-1, make_seq([1, 2, 3, 4]), 0) is not None

    bm.deallocate(seq1)                         # bundle 名下的块回到 free list
    while bm.free_block_ids:                    # 池耗尽:逐出所有残留 bundle
        bm.allocate_block()
    assert bm.search_linear_states(-1, make_seq([1, 2, 3, 4]), 0) is None
