from collections import deque

from kavor.config import Config
from kavor.engine.sequence import Sequence, SequenceStatus
from kavor.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.config = config
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size,
            has_linear_attention=config.has_linear_attention,
            num_state_groups=config.num_linear_state_groups)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)
        
    def schedule(self) -> tuple[list[Sequence], bool]:
        if self.config.has_linear_attention:
            return self.schedule_linear_attention()
        else:
            return self.schedule_full_attention()
    
    def schedule_linear_attention(self) -> tuple[list[Sequence], bool]:
            scheduled_seqs = []
            num_batched_tokens = 0
    
            # prefill
            skipped = 0
            while (self.waiting and len(scheduled_seqs) < self.max_num_seqs
                    and skipped < len(self.waiting)):
                seq = self.waiting[0]
                remaining = self.max_num_batched_tokens - num_batched_tokens
                if remaining == 0:
                    break
                if not seq.block_table:
                    num_cached_blocks, bundle = self.block_manager.can_allocate_prefill(seq)
                    if num_cached_blocks == -1:
                        break
                    if bundle and bundle.partial_token_ids:
                        num_tokens = seq.num_tokens - (num_cached_blocks * self.block_size + len(bundle.partial_token_ids))
                    else:
                        num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
                else:
                    num_tokens = seq.num_tokens - seq.num_cached_tokens
                if remaining < num_tokens and (scheduled_seqs or seq.is_multimodal):
                    break
                if not seq.block_table:
                    self.block_manager.allocate_prefill(seq, num_cached_blocks, bundle)
                    seq.stats.mark_prefill_start()
                    if self.block_manager.enable_prefix_cache:
                        seq.stats.record_prefix_cache(seq.num_cached_tokens)
                seq.num_scheduled_tokens = self.clamp_tokens(seq, num_tokens, remaining)
                            
                print(f"num_tokens: {seq.num_tokens}")
                print(f"num_scheduled_tokens: {seq.num_scheduled_tokens}")
                
                num_batched_tokens += seq.num_scheduled_tokens
                if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                    seq.status = SequenceStatus.RUNNING             # 毕业:出队进 running，准备 decode
                    self.waiting.popleft()
                    self.running.append(seq)
                elif seq.num_scheduled_tokens < remaining:
                    self.waiting.rotate(-1)                         # 主动停在快照边界:让路,本步不再回访
                    skipped += 1
                else:
                    pass                                            # 预算耗尽:留在队头,下步优先续算
                scheduled_seqs.append(seq)
    
            if scheduled_seqs:
                return scheduled_seqs, True
    
            # decode
            while self.running and len(scheduled_seqs) < self.max_num_seqs:
                seq = self.running.popleft()
                # 如果 seq 不够 block，就从后面的 seq 抢占
                while not self.block_manager.can_append_block(seq):
                    if self.running:
                        self.preempt(self.running.pop())
                    else:
                        self.preempt(seq)
                        break
                else:
                    seq.num_scheduled_tokens = 1
                    seq.is_prefill = False
                    self.block_manager.may_append_block(seq)
                    scheduled_seqs.append(seq)
            assert scheduled_seqs
            self.running.extendleft(reversed(scheduled_seqs))
            return scheduled_seqs, False

    def schedule_full_attention(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks, bundle = self.block_manager.can_allocate_prefill(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and (scheduled_seqs or seq.is_multimodal):
                break
            if not seq.block_table:
                self.block_manager.allocate_prefill(seq, num_cached_blocks, bundle)
                seq.stats.mark_prefill_start()
                if self.block_manager.enable_prefix_cache:
                    seq.stats.record_prefix_cache(seq.num_cached_tokens)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            # 如果 seq 不够 block，就从后面的 seq 抢占
            while not self.block_manager.can_append_block(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append_block(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                if seq.store_linear_states:
                    self.block_manager.record_linear_cached_blocks(seq)
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                
                
    def clamp_tokens(self, seq: Sequence, num_tokens: int, remaining: int):
        """
            1. clamp scheduled tokens for linear states snapshot
            2. it will cause more prefill stages for one sequence
        """
        snap_end_token_id = 248068  # specialize for Qwen3.5
        if seq.token_ids[seq.num_cached_tokens] == snap_end_token_id:
            return min(num_tokens, remaining)
        elif snap_end_token_id not in seq.token_ids[seq.num_cached_tokens + 1:]:
            return min(num_tokens, remaining)
        else:
            seq.store_linear_states = True
            snap_end_token_pos = seq.token_ids.index(snap_end_token_id)
            return min(snap_end_token_pos - seq.num_cached_tokens, num_tokens, remaining)
