import os
from dataclasses import dataclass

from kavor.models.configuration_auto import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    trust_remote_code: bool = False
    hf_config: AutoConfig | None = None
    text_config: AutoConfig = None,
    vision_config: AutoConfig = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        if not os.path.isdir(self.model):
            raise RuntimeError(f"Model not found: {self.model}")
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if self.hf_config.vision_config:
            self.text_config = self.hf_config.text_config
            self.vision_config = self.hf_config.vision_config
        else:
            self.text_config = self.hf_config
        max_position_embeddings = self.text_config.max_position_embeddings
        self.max_model_len = min(self.max_model_len, max_position_embeddings)