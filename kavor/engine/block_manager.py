import hashlib
import pickle
from collections import deque

from kavor.engine.sequence import Sequence
from kavor.utils.logger import logger

NONE_HASH = -1


def resident_blocks_per_seq(num_kv_blocks: int, num_state_groups: int) -> int:
    return num_kv_blocks + num_state_groups


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
        self.linear_hash_to_block_id: dict[int, list[int]] = dict()
        self.has_linear_attention = has_linear_attention
        self.num_state_groups = num_state_groups if has_linear_attention else 0
        # self.enable_prefix_cache = not has_linear_attention
        self.enable_prefix_cache = True
        self.padding_block_id = num_blocks - 1 if has_linear_attention else None
        free_end = num_blocks - 1 if has_linear_attention else num_blocks
        assert free_end > 0, (
            f"页池只有 {num_blocks} 块,不够给线性注意力留一个占位块:"
            "调大 gpu_memory_utilization 或减小 max_model_len")
        self.free_block_ids: deque[int] = deque(range(free_end))
        self.used_block_ids: set[int] = set()
        # prefix cache 命中统计:每个请求开始 prefill 时累计一次
        self.prefix_cache_hit_tokens = 0
        self.prefix_cache_prompt_tokens = 0
        # chunk 快照可见性:冻结过的快照数 / 因缺页没留快照的边界数
        self.linear_state_cached = 0
        self.linear_state_skipped = 0
        self.linear_state_shortage_warned = False

    @classmethod
    def hash_block_tokens(cls, parent_block_hash: int, curr_block_token_ids: list[int], extra_keys=None):
        if not parent_block_hash:
            parent_block_hash = NONE_HASH
        curr_block_token_ids_tuple = tuple(curr_block_token_ids)
        hush_group = (parent_block_hash, curr_block_token_ids_tuple, extra_keys)
        return hashlib.sha256(pickle.dumps(hush_group, protocol=5)).digest()

    def allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1:
            if self.hash_to_block_id.get(block.hash) == block_id:
                del self.hash_to_block_id[block.hash]
            state_ids = self.linear_hash_to_block_id.get(block.hash)
            if state_ids is not None and block_id in state_ids:
                del self.linear_hash_to_block_id[block.hash]
                for other in state_ids:
                    self.blocks[other].hash = -1
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def _hold_block(self, block_id: int):
        """让一条序列持有一块页:已被别的序列持有就加引用,否则从 free list 里拿回来。"""
        block = self.blocks[block_id]
        if block_id in self.used_block_ids:
            block.ref_count += 1
        else:
            block.ref_count = 1
            self.free_block_ids.remove(block_id)
            self.used_block_ids.add(block_id)

    def _release_block(self, block_id: int):
        """引用减一;归零的页回到 free list 尾部(最近用过,最后被逐出)。"""
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self.deallocate_block(block_id)

    def can_allocate_prefill(self, seq: Sequence):
        num_cached_blocks = 0
        state_ids = None
        num_needed_blocks = seq.num_blocks

        if self.enable_prefix_cache:
            prefix_hash_id = NONE_HASH
            chain = []
            for i in range(seq.num_blocks - 1):
                token_ids = seq.block(i)
                prefix_hash_id = self.hash_block_tokens(prefix_hash_id, token_ids, seq.block_media_hashes(i))
                block_id = self.hash_to_block_id.get(prefix_hash_id, -1)
                if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                    break
                chain.append((prefix_hash_id, block_id))
            if self.has_linear_attention:
                for i in range(len(chain) - 1, -1, -1):
                    state_ids = self.linear_hash_to_block_id.get(chain[i][0])
                    if state_ids is not None:
                        num_cached_blocks = i + 1
                        break
            else:
                num_cached_blocks = len(chain)
            num_needed_blocks -= sum(1 for _, block_id in chain[:num_cached_blocks] if block_id in self.used_block_ids)
        num_needed_blocks = resident_blocks_per_seq(num_needed_blocks, self.num_state_groups)
        # lack of enough blocks for prefill
        if len(self.free_block_ids) < num_needed_blocks:
            return -1, None
        return num_cached_blocks, state_ids

    def allocate_prefill(self, seq: Sequence, num_cached_blocks: int, state_ids: list[int] | None):
        assert not seq.block_table
        
        if state_ids is not None:
            for state_block_id in state_ids:
                self._hold_block(state_block_id)
            seq.state_copy_src = list(state_ids)

        # KV Cache block reuse
        prefix_hash_id = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            prefix_hash_id = self.hash_block_tokens(prefix_hash_id, token_ids, seq.block_media_hashes(i))
            block_id = self.hash_to_block_id[prefix_hash_id]
            self._hold_block(block_id)
            seq.block_table.append(block_id)
        cached_tokens = num_cached_blocks * self.block_size

        # allocate new KV blocks
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self.allocate_block())

        # allocate state blocks
        if self.has_linear_attention:
            assert seq.state_block_ids is None
            seq.state_block_ids = [self.allocate_block() for _ in range(self.num_state_groups)]
        seq.num_cached_tokens = cached_tokens
        if self.enable_prefix_cache:
            self.prefix_cache_hit_tokens += cached_tokens
            self.prefix_cache_prompt_tokens += seq.num_tokens

    def cache_linear_state(self, seq: Sequence, rotate: bool = True):
        if not self.has_linear_attention or seq.num_cached_tokens % self.block_size:
            return
        parent_hash = self.blocks[seq.block_table[seq.num_cached_tokens // self.block_size - 1]].hash
        if parent_hash in self.linear_hash_to_block_id:
            return
        new_ids = None
        if rotate:
            if len(self.free_block_ids) < self.num_state_groups:
                self.linear_state_skipped += 1
                if not self.linear_state_shortage_warned:
                    self.linear_state_shortage_warned = True
                    logger.warning(
                        "页池没有空闲块,chunk 边界上的线性状态快照被跳过(序列仍可正常续算,"
                        "只是这个边界不进前缀缓存)。需要命中率时,把 --max-num-seqs 调到容量以下,"
                        "给快照留出空闲页")
                return
            new_ids = [self.allocate_block() for _ in range(self.num_state_groups)]
        frozen = list(seq.state_block_ids)
        self.linear_hash_to_block_id[parent_hash] = frozen
        for block_id in frozen:
            self.blocks[block_id].hash = parent_hash
        self.linear_state_cached += 1
        if rotate:
            seq.state_copy_src = frozen
            seq.state_block_ids = new_ids

    def release_state_copy(self, seq: Sequence):
        """copy 已经发射过(本序列这一步的 forward 之前):冻结的快照页还给缓存。

        归还的页进 free list 尾部,哈希表里的登记留着,别的序列照样能命中。
        """
        if seq.state_copy_src is None:
            return
        for block_id in seq.state_copy_src:
            self._release_block(block_id)
        seq.state_copy_src = None

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            self._release_block(block_id)
        if seq.state_block_ids is not None:
            for block_id in seq.state_block_ids:
                self._release_block(block_id)
            seq.state_block_ids = None
        self.release_state_copy(seq)
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
        prefix_hash_id = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            prefix_hash_id = self.hash_block_tokens(prefix_hash_id, token_ids, seq.block_media_hashes(i))
            block.update(prefix_hash_id, token_ids)
            self.hash_to_block_id[prefix_hash_id] = block.block_id
