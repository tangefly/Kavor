"""Scheduler 单元测试:full attention / linear attention 两条调度路径。

模拟引擎主循环:schedule() → postprocess(注入固定 token),不需要模型。
生成 token 固定为 5(eos=0),靠 max_tokens 终止。
"""

import pytest

from kavor.engine.sequence import SequenceStatus
from kavor.engine.scheduler import Scheduler

SNAP_TOKEN = 248068   # clamp_tokens 中 Qwen3.5 的快照边界 token


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

def test_linear_full_prefill_and_state_blocks(make_seq, make_config):
    config = make_config(has_linear_attention=True, num_linear_state_groups=2,
                         num_kvcache_blocks=32)
    sched = Scheduler(config)
    seq = make_seq(list(range(9)), max_tokens=2)   # 无快照符,clamp 走 min 路径
    sched.add(seq)

    batch, is_prefill = sched.schedule()
    assert is_prefill and batch == [seq]
    assert len(seq.state_block_ids) == 2
    assert len(seq.state_cache_ids) == 2
    sched.postprocess(batch, [5], True)

    assert drive(sched) == [False]
    assert seq.is_finished
    assert seq.state_block_ids is None and seq.state_cache_ids is None
    assert len(sched.block_manager.free_block_ids) == 31   # 占位块除外全部归还


def test_linear_snapshot_boundary_clamps_prefill(make_seq, make_config):
    config = make_config(has_linear_attention=True, num_linear_state_groups=2,
                         num_kvcache_blocks=32)
    sched = Scheduler(config)
    tokens = [1, 2, 3, SNAP_TOKEN, 5, 6, 7, 8, 9]
    seq1 = make_seq(tokens, max_tokens=1)
    sched.add(seq1)

    # 第一段被 clamp 在快照符(位置 3)之前,并标记记录 linear states
    batch, is_prefill = sched.schedule()
    assert is_prefill and seq1.num_scheduled_tokens == 3
    assert seq1.store_linear_states is True
    assert list(sched.waiting) == [seq1]          # 未毕业,rotate 让路
    sched.postprocess(batch, [5], True)
    assert sched.block_manager.linear_hash_table.get(-1)   # bundle 已记录

    # 第二段越过快照符跑完 prefill
    batch, _ = sched.schedule()
    assert seq1.num_scheduled_tokens == 6
    assert seq1.status == SequenceStatus.RUNNING
    sched.postprocess(batch, [5], True)           # 毕业 + max_tokens=1 → 结束
    assert seq1.is_finished

    # 新请求共享 [1,2,3] 前缀:直接命中 bundle,partial 3 个 token 免算
    seq2 = make_seq(tokens + [10, 11], max_tokens=1)
    sched.add(seq2)
    batch, is_prefill = sched.schedule()
    assert is_prefill and batch == [seq2]
    assert seq2.num_cached_tokens == 3
    assert seq2.pending_tail_copy is not None
    assert drive(sched)
    assert seq2.is_finished


def test_clamp_tokens_unit(make_seq, make_config):
    sched = Scheduler(make_config())              # clamp 不依赖 linear 配置

    # 快照符在当前位置之后:截断到快照符前,并标记存状态
    seq = make_seq([9, SNAP_TOKEN, 3, 4, 5])
    assert sched.clamp_tokens(seq, 5, 10) == 1
    assert seq.store_linear_states is True
    # 预算不够到达快照符:按预算截断,未到边界不标记
    seq = make_seq([9, 8, SNAP_TOKEN, 4, 5])
    assert sched.clamp_tokens(seq, 5, 1) == 1
    assert seq.store_linear_states is False

    # 当前位置就是快照符:不截断
    seq = make_seq([SNAP_TOKEN, 2, 3])
    assert sched.clamp_tokens(seq, 3, 10) == 3

    # 没有快照符:退化为 min(num_tokens, remaining)
    seq = make_seq([1, 2, 3, 4])
    assert sched.clamp_tokens(seq, 4, 2) == 2
