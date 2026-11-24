"""Scheduler 单元测试:full attention / linear attention 两条调度路径。

模拟引擎主循环:schedule() → postprocess(注入固定 token),不需要模型。
生成 token 固定为 5(eos=0),靠 max_tokens 终止。
"""

import pytest

from kavor.engine.sequence import SequenceStatus
from kavor.engine.scheduler import Scheduler

THINK_TOKEN = 248068   # Qwen3.5 的 <think>:现在只是普通 token,不再触发任何快照


def drive(scheduler, token=5, max_steps=1000):
    """跑到所有序列结束,返回每步的 is_prefill 标记。"""
    steps = []
    while not scheduler.is_finished():
        batch, is_prefill = scheduler.schedule()
        assert batch
        scheduler.postprocess(batch, [token] * len(batch), is_prefill)
        steps.append(is_prefill)
        assert len(steps) < max_steps, "调度器未收敛"
    return steps


def prefill(scheduler, seq, token=5):
    """把一条序列的 prefill 跑到毕业,返回每段 prefill 的 token 数。"""
    chunks = []
    while seq.status == SequenceStatus.WAITING:
        batch, is_prefill = scheduler.schedule()
        assert is_prefill and batch == [seq]
        chunks.append(seq.num_scheduled_tokens)
        scheduler.postprocess(batch, [token], True)
    return chunks


# ---------------------------------------------------------------- full attention

def test_single_seq_prefill_then_decode(make_seq, make_config):
    sched = Scheduler(make_config())
    seq = make_seq(list(range(6)), max_tokens=3)
    sched.add(seq)

    batch, is_prefill = sched.schedule()
    assert is_prefill and batch == [seq]
    assert seq.num_scheduled_tokens == 6
    assert seq.status == SequenceStatus.RUNNING
    assert list(sched.running) == [seq]
    sched.postprocess(batch, [5], True)         # prefill 收尾补一个生成 token
    assert seq.num_tokens == 7 and seq.num_completion_tokens == 1

    batch, is_prefill = sched.schedule()
    assert not is_prefill and batch == [seq]
    assert seq.num_scheduled_tokens == 1 and not seq.is_prefill
    assert drive(sched) == [False, False]        # 补齐 max_tokens=3 的两步 decode


def test_chunked_prefill_splits_by_token_budget(make_seq, make_config):
    sched = Scheduler(make_config(max_num_batched_tokens=4))
    seq = make_seq(list(range(10)), max_tokens=2)
    sched.add(seq)

    batch, is_prefill = sched.schedule()         # 第 1 段:4 个 token
    assert is_prefill and seq.num_scheduled_tokens == 4
    sched.postprocess(batch, [5], True)          # 分段 prefill 不补生成 token
    assert seq.num_cached_tokens == 4 and seq.num_tokens == 10
    assert list(sched.waiting) == [seq]          # 没毕业,留在队头继续

    batch, _ = sched.schedule()                  # 第 2 段:4 个 token
    assert seq.num_scheduled_tokens == 4
    sched.postprocess(batch, [5], True)
    assert seq.num_cached_tokens == 8

    batch, _ = sched.schedule()                  # 第 3 段:收尾 2 个 token
    assert seq.num_scheduled_tokens == 2
    assert seq.status == SequenceStatus.RUNNING
    sched.postprocess(batch, [5], True)          # 毕业段补生成 token
    assert seq.num_completion_tokens == 1

    assert drive(sched) == [False]               # decode 到 max_tokens=2 结束


def test_prefill_batches_multiple_seqs(make_seq, make_config):
    sched = Scheduler(make_config(max_num_batched_tokens=32))
    seqs = [make_seq(list(range(i, i + 6)), max_tokens=2) for i in range(3)]
    for seq in seqs:
        sched.add(seq)

    batch, is_prefill = sched.schedule()
    assert is_prefill and batch == seqs
    assert all(seq.status == SequenceStatus.RUNNING for seq in seqs)
    sched.postprocess(batch, [5] * len(batch), True)
    assert drive(sched) == [False]               # 一个 decode 步齐收尾


def test_max_num_seqs_limits_batch(make_seq, make_config):
    sched = Scheduler(make_config(max_num_seqs=1, max_num_batched_tokens=32))
    seq1 = make_seq([1, 2, 3, 4], max_tokens=2)
    seq2 = make_seq([5, 6, 7, 8], max_tokens=2)
    sched.add(seq1)
    sched.add(seq2)

    _, is_prefill = sched.schedule()
    assert is_prefill                            # 只排得下 seq1 的 prefill
    sched.postprocess([seq1], [5], True)

    batch, is_prefill = sched.schedule()         # seq2 prefill
    assert is_prefill and batch == [seq2]
    sched.postprocess(batch, [5], True)

    batch, is_prefill = sched.schedule()         # decode 同样受 max_num_seqs 限制
    assert not is_prefill and batch == [seq1]
    sched.postprocess(batch, [5], False)
    assert drive(sched) == [False]


def test_head_of_line_blocking_on_budget(make_seq, make_config):
    sched = Scheduler(make_config(max_num_batched_tokens=10))
    seq1 = make_seq(list(range(6)), max_tokens=2)
    seq2 = make_seq(list(range(10, 16)), max_tokens=2)
    sched.add(seq1)
    sched.add(seq2)

    batch, is_prefill = sched.schedule()
    # seq1 用掉 6 预算后,seq2 还差 2 个 token 且已有人排队 → 让路
    assert is_prefill and batch == [seq1]
    sched.postprocess(batch, [5], True)

    assert drive(sched)                          # seq2 随后单独 prefill 再 decode
    assert seq1.is_finished and seq2.is_finished


def test_preempt_on_block_exhaustion(make_seq, make_config):
    """块耗尽时从 running 尾部抢占:资源回收、回到 waiting 队头重算。"""
    sched = Scheduler(make_config(num_kvcache_blocks=4, max_num_batched_tokens=64))
    seq1 = make_seq([10], max_tokens=12)
    seq2 = make_seq([20], max_tokens=12)
    sched.add(seq1)
    sched.add(seq2)

    preempted = []
    real_preempt = sched.preempt

    def spy(seq):
        preempted.append(seq)
        real_preempt(seq)

    sched.preempt = spy
    while not sched.is_finished():
        batch, is_prefill = sched.schedule()
        sched.postprocess(batch, [5] * len(batch), is_prefill)
        if preempted:
            break

    assert preempted == [seq2]                   # 队尾的 seq2 被牺牲
    assert seq2.status == SequenceStatus.WAITING
    assert seq2.block_table == []                # 块已全部回收
    assert seq2.state_block_ids is None
    assert list(sched.waiting)[0] is seq2        # 回到队头等待重算

    drive(sched)                                 # seq1 先完成,seq2 重新 prefill
    assert seq1.is_finished and seq2.is_finished
    assert len(sched.block_manager.free_block_ids) == 4


def test_decode_finishes_on_eos(make_seq, make_config):
    sched = Scheduler(make_config())             # eos=0
    seq = make_seq([1, 2, 3, 4], max_tokens=100)
    sched.add(seq)
    batch, _ = sched.schedule()
    sched.postprocess(batch, [5], True)          # prefill 毕业,补生成 token

    batch, _ = sched.schedule()                  # decode
    assert list(sched.running) == [seq]
    sched.postprocess(batch, [0], False)         # 生成 eos
    assert seq.is_finished
    assert not sched.running and sched.is_finished()
    assert len(sched.block_manager.free_block_ids) == 64   # 块全部归还


def test_finishes_on_max_tokens(make_seq, make_config):
    sched = Scheduler(make_config())
    seq = make_seq(list(range(6)), max_tokens=3)
    sched.add(seq)
    assert drive(sched) == [True, False, False]  # prefill 补 1 + decode 补 2
    assert seq.num_completion_tokens == 3
    assert seq.is_finished and sched.is_finished()


# ---------------------------------------------------------------- linear attention

def linear_scheduler(make_config, **overrides):
    values = dict(has_linear_attention=True, num_linear_state_groups=2)
    values.update(overrides)
    return Scheduler(make_config(**values))


def test_linear_full_prefill_and_state_blocks(make_seq, make_config):
    sched = linear_scheduler(make_config, num_kvcache_blocks=32)
    seq = make_seq(list(range(3)), max_tokens=2)   # 不足一个 chunk:一段算完
    sched.add(seq)

    batch, is_prefill = sched.schedule()
    assert is_prefill and batch == [seq]
    assert len(seq.state_block_ids) == 2
    assert seq.state_copy_src is None            # 从头算,没有快照要 copy
    sched.postprocess(batch, [5], True)

    assert drive(sched) == [False]
    assert seq.is_finished
    assert seq.state_block_ids is None and seq.state_copy_src is None
    assert len(sched.block_manager.free_block_ids) == 31   # 占位块除外全部归还


def test_linear_prefill_stops_on_chunk_boundaries(make_seq, make_config):
    """每段 prefill 停在 block_size 的整倍数上:线性状态只有在 chunk 边界才留得下快照。"""
    sched = linear_scheduler(make_config)
    bm = sched.block_manager
    seq = make_seq(list(range(9)), max_tokens=1)
    sched.add(seq)

    assert prefill(sched, seq) == [4, 4, 1]      # 两个整 chunk + prompt 结尾
    assert seq.is_finished
    assert len(bm.linear_hash_to_block_id) == 2  # 4 和 8 两处都留下了快照
    assert len(bm.free_block_ids) == 63


def test_linear_prefill_realigns_to_the_boundary_on_a_small_budget(make_seq, make_config):
    """预算不够走到边界就先用完预算,下一段从块中间起步、到边界再停。"""
    sched = linear_scheduler(make_config, max_num_batched_tokens=3)
    seq = make_seq(list(range(10)), max_tokens=1)
    sched.add(seq)

    assert prefill(sched, seq) == [3, 1, 3, 1, 2]   # 3 | 1(到 4) | 3 | 1(到 8) | 2(结尾)
    assert len(sched.block_manager.linear_hash_to_block_id) == 2


def test_think_token_no_longer_splits_the_prefill(make_seq, make_config):
    sched = linear_scheduler(make_config)
    seq = make_seq([1, 2, THINK_TOKEN, 3, 4, 5], max_tokens=1)
    sched.add(seq)

    assert prefill(sched, seq) == [4, 2]         # 不是停在 <think> 前面的 2


def test_repeated_prompt_resumes_from_the_last_chunk_boundary(make_seq, make_config):
    sched = linear_scheduler(make_config)
    bm = sched.block_manager
    tokens = list(range(1, 15))
    seq1 = make_seq(tokens, max_tokens=1)
    sched.add(seq1)
    assert drive(sched) and seq1.is_finished
    assert len(bm.linear_hash_to_block_id) == 3  # 4、8、12 三个边界

    seq2 = make_seq(tokens + [50, 51], max_tokens=1)   # 同前缀,多 2 个 token
    sched.add(seq2)
    batch, is_prefill = sched.schedule()
    assert is_prefill and batch == [seq2]
    assert seq2.num_cached_tokens == 12          # 3 个整 chunk;最后 2 个 token 不复用
    assert seq2.state_copy_src is not None       # 从快照 copy 起步
    assert seq2.stats.prefix_cache_hit_tokens == 12
    assert seq2.num_scheduled_tokens == 4        # 只算 12..16
    assert drive(sched) and seq2.is_finished
    assert len(bm.free_block_ids) == 63


def test_decode_boundary_freezes_state_and_swaps_in_a_new_block(make_seq, make_config):
    """decode 原地更新状态块;走到 chunk 边界时冻结 + 换块,下一步 forward 前 copy。"""
    sched = linear_scheduler(make_config)
    bm = sched.block_manager
    seq = make_seq(list(range(5)), max_tokens=8)   # 5 + 8 = 13 个 token,decode 跨过 8 和 12
    sched.add(seq)
    assert prefill(sched, seq) == [4, 1]
    assert seq.state_copy_src is None             # 第一个边界的 copy 已随第二段 prefill 发完
    first = bm.linear_hash_to_block_id[bm.blocks[seq.block_table[0]].hash]   # 4 处的快照
    working = list(seq.state_block_ids)           # 接过 4 处状态的可变块
    assert not set(working) & set(first)
    assert all(b in bm.free_block_ids for b in first)          # 已还给缓存,登记还在
    # 一直 decode 到可变块停在下一个边界(8)上
    while seq.state_copy_src is None:
        batch, is_prefill = sched.schedule()
        assert not is_prefill and batch == [seq]
        sched.postprocess(batch, [5], False)
    assert seq.num_tokens - 1 == 8                # 状态块停在 8 个 token 上
    assert bm.linear_hash_to_block_id[bm.blocks[seq.block_table[1]].hash] == working
    assert seq.state_copy_src == working          # 下一步 decode 的 forward 前 copy 进新块
    assert not set(seq.state_block_ids) & set(working)
    assert all(b not in bm.free_block_ids for b in working)

    # copy 发完的那一步:旧块归还缓存,登记留着
    batch, is_prefill = sched.schedule()
    sched.postprocess(batch, [5], False)
    assert seq.state_copy_src is None
    assert all(b in bm.free_block_ids for b in working)
    assert bm.linear_hash_to_block_id[bm.blocks[seq.block_table[1]].hash] == working

    assert drive(sched) and seq.is_finished
    assert len(bm.free_block_ids) == 63


def test_next_turn_hits_the_state_frozen_during_decode(make_seq, make_config):
    """多轮对话:上一轮在 decode 阶段跨过 chunk 边界,下一轮的 prompt 就能从这里续算。

    上一轮的 prompt 不足一个 chunk,prefill 阶段只留得下 4 这一处快照;8 这一处全靠
    decode 跨边界时冻结。
    """
    sched = linear_scheduler(make_config)
    seq1 = make_seq(list(range(5)), max_tokens=4)   # 5 + 4 = 9 个 token,decode 跨过 8
    sched.add(seq1)
    assert drive(sched) and seq1.is_finished
    history = seq1.token_ids                      # 下一轮的 prompt = 上一轮的全部 token
    assert len(history) == 9

    seq2 = make_seq(history + [30, 31], max_tokens=1)   # 11 个 token:2 个整块 + 3
    sched.add(seq2)
    batch, is_prefill = sched.schedule()
    assert is_prefill and batch == [seq2]
    assert seq2.num_cached_tokens == 8
    assert seq2.num_scheduled_tokens == 3
    assert drive(sched) and seq2.is_finished


def test_preempt_with_a_pending_copy_frees_every_block(make_seq, make_config):
    """刚冻结、copy 还没发时被抢占:冻结的页留给缓存,其余全部归还,重新准入时照常命中。"""
    sched = linear_scheduler(make_config, num_kvcache_blocks=10)   # 9 个可用块
    bm = sched.block_manager
    seq = make_seq(list(range(8)), max_tokens=6)
    sched.add(seq)
    assert prefill(sched, seq) == [4, 4]          # prompt 正好在边界上算完
    snapshot = list(seq.state_copy_src)           # 8 处的快照;copy 要到下一步 decode 的 forward 前才发
    assert snapshot is not None
    assert len(bm.free_block_ids) == 3            # 2 KV + 2 状态 + 2 待 copy 的快照

    # 这一步还没轮到它就被抢占(调度器只对 running 里的序列调用 preempt)
    sched.running.remove(seq)
    sched.preempt(seq)
    assert seq.state_copy_src is None and seq.state_block_ids is None
    assert seq.block_table == []
    assert all(bm.blocks[b].ref_count == 0 for b in range(9))
    assert len(bm.free_block_ids) == 9
    assert len(bm.linear_hash_to_block_id) == 2   # 两组快照都留给缓存

    # 重新准入:它已经有 9 个 token,前两个整块的 KV 和 8 处的状态快照都还在缓存里,
    # 从 8 个 token 处续算(刚生成的那个 token 还没进 KV,本来就要重算)
    batch, is_prefill = sched.schedule()
    assert is_prefill and batch == [seq]
    assert seq.num_cached_tokens == 8 and seq.state_copy_src == snapshot
    assert drive(sched) and seq.is_finished
    assert len(bm.free_block_ids) == 9


def test_linear_decode_holds_only_resident_blocks(make_seq, make_config):
    """无快照符的负载:每条序列只占 1 个 KV 块 + num_state_groups 个状态块。"""
    sched = linear_scheduler(make_config, num_kvcache_blocks=16, max_num_batched_tokens=256)
    seqs = [make_seq([3 * i, 3 * i + 1, 3 * i + 2], max_tokens=2) for i in range(5)]
    for seq in seqs:
        sched.add(seq)

    batch, is_prefill = sched.schedule()
    assert is_prefill and batch == seqs
    assert all(seq.state_copy_src is None for seq in seqs)
    # 15 个可用块(16 减占位块)/ 3 = 5,正好装下,没有为快照多留一倍
    assert not sched.block_manager.free_block_ids
    assert drive(sched)
    assert all(seq.is_finished for seq in seqs)
    assert len(sched.block_manager.free_block_ids) == 15


def test_full_pool_skips_the_snapshot_but_sequences_still_finish(make_seq, make_config):
    """池满时边界不留快照,序列在原块上原地继续,不会被卡住。"""
    sched = linear_scheduler(make_config, num_kvcache_blocks=7)   # 6 个可用块
    bm = sched.block_manager
    # 3 个 KV 块 + 2 个状态块,只剩 1 个空闲:凑不出下一组状态块(要 2 个)
    # 12 + 4 = 16 个 token,正好是 4 个 KV 块,decode 时还能拿到第 4 个 KV 块
    seq = make_seq(list(range(12)), max_tokens=4)
    sched.add(seq)

    assert drive(sched)
    assert seq.is_finished and seq.num_tokens == 16
    assert bm.linear_state_skipped == 3           # 4、8、12 三处边界都没留下快照
    assert bm.linear_hash_to_block_id == {}
    assert len(bm.free_block_ids) == 6


def test_clamp_tokens_unit(make_seq, make_config):
    sched = linear_scheduler(make_config)

    # 从块中间起步:截到下一个 chunk 边界
    seq = make_seq([9, 8, 7, 6, 5, 4])
    seq.num_cached_tokens = 1
    assert sched.clamp_tokens(seq, 5, 10) == 3
    # 正好在边界上,剩余不止一个 chunk:一次一个 chunk
    seq.num_cached_tokens = 4
    assert sched.clamp_tokens(seq, 9, 10) == 4
    # prompt 在边界内结束 / 预算不够走到边界:取更短的那个
    assert sched.clamp_tokens(seq, 2, 10) == 2
    assert sched.clamp_tokens(seq, 9, 1) == 1
