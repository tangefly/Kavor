from copy import copy
from enum import Enum, auto
from itertools import count
import torch
from dataclasses import dataclass

from kavor.sampling_params import SamplingParams
from .multimodal import MultiModalInputs

@dataclass
class MultiModalVisionData:
    mm_token_type_ids: list[int] | None = None
    pixel_values: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None
    pixel_values_videos: torch.Tensor | None = None
    video_grid_thw: torch.Tensor | None = None
    rope_delta: int = 0


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, mm_inputs: MultiModalInputs, sampling_params = SamplingParams()):
        
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        
        self.token_ids = copy(mm_inputs.token_ids)
        self.mm = MultiModalVisionData(
            mm_token_type_ids=mm_inputs.mm_token_type_ids,
            pixel_values=mm_inputs.pixel_values,
            image_grid_thw=mm_inputs.image_grid_thw,
            pixel_values_videos=mm_inputs.pixel_values_videos,
            video_grid_thw=mm_inputs.video_grid_thw,
            rope_delta=mm_inputs.rope_delta
        )
        
        self.last_token = self.token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(self.token_ids)
        
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        
        self.first_token_time = None
        self.finish_time = None

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

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        mm = None if self.mm is None else self.mm.for_workers(prefill=self.is_prefill)
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state, mm)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state, mm = state
        self.mm = mm
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
