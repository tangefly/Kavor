import os
import socket
from dataclasses import dataclass

from kavor.models.configuration_auto import AutoConfig


def free_port() -> int:
    if env := os.environ.get("KAVOR_RDZV_PORT"):
        return int(env)
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    enable_mm_cache: bool = False
    mm_cache_size_gb: float = 1.0
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    trust_remote_code: bool = False
    hf_config: AutoConfig | None = None
    text_config: AutoConfig = None,
    vision_config: AutoConfig = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    rdzv_port: int = -1
    has_linear_attention: bool = False
    num_linear_state_groups: int = 0

    def __post_init__(self):
        if not os.path.isdir(self.model):
            raise RuntimeError(f"Model not found: {self.model}")
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.mm_cache_size_gb >= 0, "mm_cache_size_gb 不能为负"
        self.rdzv_port = self.rdzv_port if self.rdzv_port > 0 else free_port()
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if getattr(self.hf_config, "vision_config", None):
            self.text_config = self.hf_config.text_config
            self.vision_config = self.hf_config.vision_config
        else:
            self.text_config = self.hf_config
        max_position_embeddings = self.text_config.max_position_embeddings
        self.max_model_len = min(self.max_model_len, max_position_embeddings)