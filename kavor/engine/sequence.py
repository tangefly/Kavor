from copy import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count
from time import perf_counter

from kavor.sampling_params import SamplingParams
from .multimodal import PromptInputs


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class SeqStats:
    """一条请求的缓存/时延统计,只在引擎侧(rank0)的 Sequence 上收集。

    Sequence 发往 TP worker 时不带走这个对象(见 __getstate__),worker 上
    seq.stats 为 None。
    """

    arrival_time: float = field(default_factory=perf_counter)
    prefill_start_time: float | None = None
    first_token_time: float | None = None
    prefix_cache_hit_tokens: int | None = None
    mm_hit_items: int = 0
    mm_total_items: int = 0
    mm_hit_rows: int = 0
    mm_total_rows: int = 0
    mm_seen: set[tuple[bytes, int]] = field(default_factory=set)

    def mark_prefill_start(self):
        """记录第一次被调度做 prefill 的时刻(抢占重算不覆盖)。"""
        if self.prefill_start_time is None:
            self.prefill_start_time = perf_counter()

    def record_prefix_cache(self, hit_tokens: int):
        """记录首次分配时的 prefix cache 命中(抢占重算不覆盖)。"""
        if self.prefix_cache_hit_tokens is None:
            self.prefix_cache_hit_tokens = hit_tokens

    def record_mm_item(self, hash_id: bytes, pixel_start: int, rows: int, hit: bool):
        """按媒体出现位置去重累计:分块 prefill 重复看到同一项只算一次,
        同一请求里两张相同的图仍各算一项。"""
        key = (hash_id, pixel_start)
        if key in self.mm_seen:
            return
        self.mm_seen.add(key)
        self.mm_total_items += 1
        self.mm_total_rows += rows
        if hit:
            self.mm_hit_items += 1
            self.mm_hit_rows += rows


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, inputs: PromptInputs, sampling_params = SamplingParams()):

        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING

        self.token_ids = copy(inputs.token_ids)
        self.mm_inputs = copy(inputs.mm_inputs)

        self.last_token = self.token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(self.token_ids)
        
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.state_block_ids: list[int] | None = None
        
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

        self.request_id: str | None = None
        self.stats = SeqStats()

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size
    
    @property
    def is_multimodal(self):
        inputs = self.mm_inputs
        if inputs is None:
            return False
        if inputs.image is not None or inputs.video is not None:
            return True
        return bool(inputs.mm_token_type_ids) and any(inputs.mm_token_type_ids)

    @property
    def mm_hash_id(self) -> bytes | int:
        """媒体摘要(prefix cache 的块哈希要用);纯文本序列没有 mm_inputs,用 -1。"""
        return -1 if self.mm_inputs is None else self.mm_inputs.mm_hash_id

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        mm_inputs = None if self.mm_inputs is None else self.mm_inputs.for_workers(prefill=self.is_prefill)
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state, mm_inputs, self.state_block_ids)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state, mm_inputs, self.state_block_ids = state
        self.mm_inputs = mm_inputs
        self.stats = None       # 统计只在引擎侧收集,worker 副本不需要
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
