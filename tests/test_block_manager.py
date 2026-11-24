"""BlockManager 单元测试:块分配、prefix cache 命中/复用、linear attention 的 chunk 状态快照。

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


# ---------------------------------------------------------------- linear attention chunk state

def _linear_bm(num_blocks: int = 24) -> BlockManager:
    return BlockManager(num_blocks=num_blocks, block_size=TEST_BLOCK_SIZE,
                        has_linear_attention=True, num_state_groups=2)


def _run_step(bm: BlockManager, seq, num_tokens: int, rotate: bool = True):
    """模拟一步 forward 之后的 postprocess:整块哈希 → 释放上一步的 copy 源 → 推进 → 过边界就冻结。"""
    seq.num_scheduled_tokens = num_tokens
    bm.hash_blocks(seq)
    bm.release_state_copy(seq)
    seq.num_cached_tokens += num_tokens
    seq.num_scheduled_tokens = 0
    bm.cache_linear_state(seq, rotate=rotate)


def test_linear_prefill_budget_counts_only_resident_blocks(make_seq):
    # 1 个 KV 块 + 2 个可变状态块 = 3 块
    bm = BlockManager(num_blocks=4, block_size=TEST_BLOCK_SIZE,
                      has_linear_attention=True, num_state_groups=2)
    seq = make_seq([1, 2, 3])
    assert bm.can_allocate_prefill(seq) == (0, None)
    bm.allocate_prefill(seq, 0, None)
    assert len(seq.state_block_ids) == 2
    assert seq.state_copy_src is None          # 从头算,没有快照要 copy
    assert not bm.free_block_ids                # 3 块正好用完

    # 池再小一块就装不下:4 块池 - 1 占位 = 3,还差一个
    bm2 = BlockManager(num_blocks=3, block_size=TEST_BLOCK_SIZE,
                       has_linear_attention=True, num_state_groups=2)
    assert bm2.can_allocate_prefill(make_seq([1, 2, 3])) == (-1, None)


def test_chunk_boundary_freezes_state_and_rotates_blocks(make_seq):
    bm = _linear_bm()
    seq = make_seq(list(range(12)))             # 3 个整块
    bm.allocate_prefill(seq, 0, None)
    working = list(seq.state_block_ids)

    _run_step(bm, seq, 4)                       # 第一步正好停在第一个 chunk 边界
    block_hash = bm.blocks[seq.block_table[0]].hash
    assert bm.linear_hash_to_block_id == {block_hash: working}
    assert bm.linear_state_cached == 1
    # 换了一组新块接着算;旧块先留给 copy,仍被本序列持有,不会被逐出
    assert seq.state_copy_src == working
    assert not set(seq.state_block_ids) & set(working)
    assert all(bm.blocks[b].ref_count == 1 for b in working)
    assert not set(working) & set(bm.free_block_ids)

    _run_step(bm, seq, 2)                       # copy 发完的那一步;停在块中间
    assert seq.state_copy_src is None
    assert all(b in bm.free_block_ids for b in working)     # 归还到 free list 尾部
    assert all(bm.blocks[b].ref_count == 0 for b in working)
    assert bm.linear_hash_to_block_id == {block_hash: working}   # 登记还在,别人能命中
    assert bm.linear_state_cached == 1          # 块中间不产生新快照


def test_state_snapshot_is_keyed_by_the_kv_block_hash(make_seq):
    """第 k+1 个 chunk 走完后的状态,和第 k 个 KV 块描述同一段前缀:用同一个哈希登记。"""
    bm = _linear_bm()
    seq = make_seq(list(range(8)))
    bm.allocate_prefill(seq, 0, None)
    _run_step(bm, seq, 4)
    first = list(seq.state_copy_src)
    _run_step(bm, seq, 4)
    second = list(seq.state_copy_src)
    assert bm.linear_hash_to_block_id == {
        bm.blocks[seq.block_table[0]].hash: first,
        bm.blocks[seq.block_table[1]].hash: second,
    }


def test_step_ending_inside_a_chunk_leaves_no_snapshot(make_seq):
    """状态只在一段 forward 结束时写出:跨过去的中间边界上没有状态可冻结。"""
    bm = _linear_bm()
    seq = make_seq(list(range(10)))
    bm.allocate_prefill(seq, 0, None)
    _run_step(bm, seq, 10)
    assert bm.linear_hash_to_block_id == {} and bm.linear_state_cached == 0
    assert seq.state_copy_src is None
    assert len(bm.hash_to_block_id) == 2        # KV 整块照常登记


def test_repeated_prompt_resumes_on_the_chunk_boundary(make_seq):
    """命中长度 = KV 整块命中、且边界状态也命中的最长前缀;尾部不满一个 chunk 的部分不复用。"""
    bm = _linear_bm()
    tokens = list(range(10))                    # 2 个整块 + 2 个尾 token
    seq1 = make_seq(tokens)
    bm.allocate_prefill(seq1, 0, None)
    for n in (4, 4, 2):                         # 按 chunk 边界分三段算完
        _run_step(bm, seq1, n)
    kv_ids = list(seq1.block_table)
    snapshots = [bm.linear_hash_to_block_id[bm.blocks[b].hash] for b in kv_ids[:2]]   # 尾块没有哈希
    bm.deallocate(seq1)

    seq2 = make_seq(tokens + [50, 51])          # 同前缀,多 2 个 token
    assert bm.can_allocate_prefill(seq2) == (2, snapshots[1])
    bm.allocate_prefill(seq2, 2, snapshots[1])
    assert seq2.num_cached_tokens == 8          # 两个整 chunk;尾部的 2 个 token 不复用
    assert seq2.block_table[:2] == kv_ids[:2]   # KV 块一并复用
    assert seq2.state_copy_src == snapshots[1]  # 从 8 个 token 处的状态 copy 起步
    assert not set(seq2.state_block_ids) & set(snapshots[1])
    assert bm.prefix_cache_hit_tokens == 8


def test_partial_chunk_prefix_is_not_reusable(make_seq):
    """共享的前缀只有 1 个整 chunk + 2 个 token:能续算的只有第 4 个 token 处的状态。"""
    bm = _linear_bm()
    seq1 = make_seq(list(range(6)))
    bm.allocate_prefill(seq1, 0, None)
    for n in (4, 2):
        _run_step(bm, seq1, n)
    bm.deallocate(seq1)

    seq2 = make_seq(list(range(6)) + [40, 41, 42, 43])
    cached, state_ids = bm.can_allocate_prefill(seq2)
    assert cached == 1 and state_ids is not None
    bm.allocate_prefill(seq2, cached, state_ids)
    assert seq2.num_cached_tokens == 4          # [4, 5] 这 2 个 token 要重算


def _cache_two_chunks(bm: BlockManager, make_seq):
    """按 chunk 边界算完 [0..7] 并结束,留下 2 个 KV 块和 2 组边界状态快照。

    返回 (各 KV 块 id, 各边界上的快照块)。
    """
    seq = make_seq(list(range(8)))
    bm.allocate_prefill(seq, 0, None)
    for _ in range(2):
        _run_step(bm, seq, TEST_BLOCK_SIZE)
    kv_ids = list(seq.block_table)
    snapshots = [bm.linear_hash_to_block_id[bm.blocks[b].hash] for b in kv_ids]
    bm.deallocate(seq)
    return kv_ids, snapshots


def test_state_miss_disables_kv_reuse(make_seq):
    """前缀的线性状态算不出来,KV 块单独命中也没用:从头 prefill。"""
    bm = _linear_bm(num_blocks=10)              # 9 个可用块
    kv_ids, snapshots = _cache_two_chunks(bm, make_seq)
    assert len(bm.hash_to_block_id) == 2 and len(snapshots) == 2

    bm.linear_hash_to_block_id.clear()          # 等价于两组快照都已被逐出
    assert bm.can_allocate_prefill(make_seq(list(range(8)) + [9])) == (0, None)


def test_state_hit_falls_back_to_an_earlier_boundary(make_seq):
    bm = _linear_bm(num_blocks=10)
    kv_ids, snapshots = _cache_two_chunks(bm, make_seq)

    # 较新的那组快照没了(等价于被逐出):退回到第一个边界
    del bm.linear_hash_to_block_id[bm.blocks[kv_ids[1]].hash]
    assert bm.can_allocate_prefill(make_seq(list(range(8)) + [9])) == (1, snapshots[0])


def test_snapshot_hit_is_shared_not_stolen(make_seq):
    """同前缀的请求可以先后/同时命中同一组快照:各自 copy 一份,登记一直在。"""
    bm = _linear_bm()
    kv_ids, snapshots = _cache_two_chunks(bm, make_seq)
    seq_a = make_seq(list(range(8)) + [40])
    seq_b = make_seq(list(range(8)) + [70, 71])
    for seq in (seq_a, seq_b):
        assert bm.can_allocate_prefill(seq) == (2, snapshots[1])
        bm.allocate_prefill(seq, 2, snapshots[1])
        assert seq.state_copy_src == snapshots[1]
        assert not set(seq.state_block_ids) & set(snapshots[1])
    assert all(bm.blocks[b].ref_count == 2 for b in snapshots[1])     # 被 a、b 同时持有
    assert bm.linear_hash_to_block_id[bm.blocks[kv_ids[1]].hash] == snapshots[1]
    assert bm.prefix_cache_hit_tokens == 16

    _run_step(bm, seq_a, 1)                     # a 的 copy 发完
    assert all(bm.blocks[b].ref_count == 1 for b in snapshots[1])
    assert not set(snapshots[1]) & set(bm.free_block_ids)             # b 还在用
    _run_step(bm, seq_b, 2)
    assert all(b in bm.free_block_ids for b in snapshots[1])          # 都还给了缓存
    assert bm.linear_hash_to_block_id[bm.blocks[kv_ids[1]].hash] == snapshots[1]


def test_state_snapshot_lazily_evicted_when_its_pages_are_reused(make_seq):
    bm = _linear_bm(num_blocks=10)
    kv_ids, snapshots = _cache_two_chunks(bm, make_seq)
    first = bm.blocks[kv_ids[0]].hash
    second = bm.blocks[kv_ids[1]].hash

    while first in bm.linear_hash_to_block_id:
        bm.allocate_block()                     # 池子吃紧:从最久未用的页开始重新分配
    # 只逐出了最老的那组;同组两页一起摘掉,降级成普通空闲页
    assert bm.linear_hash_to_block_id == {second: snapshots[1]}
    assert all(bm.blocks[b].hash == -1 for b in snapshots[0])
    assert bm.hash_to_block_id == {first: kv_ids[0], second: kv_ids[1]}   # KV 条目没受影响


def test_boundary_without_free_pages_leaves_no_snapshot(make_seq):
    bm = BlockManager(num_blocks=9, block_size=TEST_BLOCK_SIZE,
                      has_linear_attention=True, num_state_groups=2)   # 8 个可用块
    seq = make_seq(list(range(12)))
    other = make_seq([50, 51])
    bm.allocate_prefill(seq, 0, None)           # 3 KV + 2 状态
    bm.allocate_prefill(other, 0, None)         # 1 KV + 2 状态 → 池满
    assert not bm.free_block_ids
    working = list(seq.state_block_ids)

    _run_step(bm, seq, 4)                       # 边界,但凑不出下一组新块
    assert bm.linear_hash_to_block_id == {}
    assert bm.linear_state_skipped == 1
    assert seq.state_block_ids == working and seq.state_copy_src is None   # 原地继续

    bm.deallocate(other)                        # 池里有页了,下一个边界照常冻结
    _run_step(bm, seq, 4)
    assert bm.linear_state_cached == 1 and seq.state_copy_src == working


def test_boundary_already_cached_by_another_sequence_is_left_alone(make_seq):
    bm = _linear_bm()
    seq1 = make_seq(list(range(12)))
    seq2 = make_seq(list(range(12)) + [99])
    bm.allocate_prefill(seq1, 0, None)
    bm.allocate_prefill(seq2, 0, None)          # 同前缀的另一条:各自从头算
    _run_step(bm, seq1, 4)
    _run_step(bm, seq2, 4)                      # 同一个边界:状态已经在表里
    assert bm.linear_state_cached == 1
    assert len(bm.linear_hash_to_block_id) == 1
    assert seq2.state_copy_src is None          # 不重复冻结,也不用换块
    assert seq2.state_block_ids != seq1.state_copy_src


def test_deallocate_releases_a_pending_copy_source(make_seq):
    """刚冻结、copy 还没发时被抢占/结束:冻结的页和新换的页都要归还。"""
    bm = _linear_bm()
    seq = make_seq(list(range(12)))
    bm.allocate_prefill(seq, 0, None)
    _run_step(bm, seq, 4)
    assert seq.state_copy_src is not None
    bm.deallocate(seq)
    assert seq.state_copy_src is None and seq.state_block_ids is None
    assert all(bm.blocks[b].ref_count == 0 for b in range(23))
    assert len(bm.free_block_ids) == 23         # 占位块除外全部归还
    assert len(bm.linear_hash_to_block_id) == 1   # 快照留给缓存


def test_finishing_on_a_boundary_keeps_the_snapshot_without_rotating(make_seq):
    bm = _linear_bm()
    seq = make_seq(list(range(8)))
    bm.allocate_prefill(seq, 0, None)
    _run_step(bm, seq, 4)
    _run_step(bm, seq, 4, rotate=False)         # prompt 正好在边界上算完,这一步就收尾
    assert bm.linear_hash_to_block_id[bm.blocks[seq.block_table[1]].hash] == seq.state_block_ids
    assert seq.state_copy_src is None           # 不用再换块
    bm.deallocate(seq)
    assert len(bm.linear_hash_to_block_id) == 2
    assert len(bm.free_block_ids) == 23
