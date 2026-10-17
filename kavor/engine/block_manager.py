from collections import deque
import xxhash
import numpy as np

from kavor.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        has_linear_attention: bool = False,
        num_state_groups: int = 0,
    ):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.has_linear_attention = has_linear_attention
        self.num_state_groups = num_state_groups if has_linear_attention else 0
        self.enable_prefix_cache = not has_linear_attention
        self.padding_block_id = num_blocks - 1 if has_linear_attention else None
        free_end = num_blocks - 1 if has_linear_attention else num_blocks
        assert free_end > 0, (
            f"页池只有 {num_blocks} 块,不够给线性注意力留一个占位块:"
            "调大 gpu_memory_utilization 或减小 max_model_len")
        self.free_block_ids: deque[int] = deque(range(free_end))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate_prefill(self, seq: Sequence) -> int:        
        hash_id = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        # prefix caching
        if self.enable_prefix_cache and seq.mm is None:
            for i in range(seq.num_blocks - 1):
                token_ids = seq.block(i)
                hash_id = self.compute_hash(token_ids, hash_id)
                block_id = self.hash_to_block_id.get(hash_id, -1)
                if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                    break
                num_cached_blocks += 1
                if block_id in self.used_block_ids:
                    num_new_blocks -= 1
        num_new_blocks += self.num_state_groups
        # lack of enough blocks for prefill
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate_prefill(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self.allocate_block())
        if self.has_linear_attention:
            assert seq.state_block_ids is None
            seq.state_block_ids = [self.allocate_block() for _ in range(self.num_state_groups)]
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self.deallocate_block(block_id)
        if seq.state_block_ids is not None:
            for block_id in seq.state_block_ids:
                self.blocks[block_id].ref_count -= 1
                self.deallocate_block(block_id)
            seq.state_block_ids = None
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append_block(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append_block(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self.allocate_block())

    def hash_blocks(self, seq: Sequence):
        if not self.enable_prefix_cache:
            return
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
