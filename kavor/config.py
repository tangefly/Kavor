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

    # Resolved from `model` in __post_init__ / by the engine.
    hf_config: AutoConfig | None = None
    text_config: AutoConfig | None = None
    vision_config: AutoConfig | None = None
    eos: int = -1

    # Batching and sequence limits.
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096

    # GPU memory.
    gpu_memory_utilization: float = 0.9
    enable_mm_cache: bool = False
    mm_cache_size_gb: float = 0.5

    # KV cache.
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    # Linear attention state cache.
    has_linear_attention: bool = False
    num_linear_state_groups: int = 0
    snapshot_decode_states: bool = False

    # Parallelism and distributed setup.
    tensor_parallel_size: int = 1
    rdzv_port: int = -1

    # Execution mode.
    enforce_eager: bool = False
    profile: bool = False

    # Model loading.
    trust_remote_code: bool = False

    def __post_init__(self):
        if not os.path.isdir(self.model):
            raise RuntimeError(f"Model not found: {self.model}")
        
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.mm_cache_size_gb >= 0, "mm_cache_size_gb must be non-negative"
        
        self.rdzv_port = self.rdzv_port if self.rdzv_port > 0 else free_port()
        self.hf_config = AutoConfig.from_pretrained(self.model)
        
        if getattr(self.hf_config, "vision_config", None):
            self.text_config = self.hf_config.text_config
            self.vision_config = self.hf_config.vision_config
        else:
            self.text_config = self.hf_config
        
        max_position_embeddings = self.text_config.max_position_embeddings
        self.max_model_len = min(self.max_model_len, max_position_embeddings)

    def __str__(self) -> str:
        """Flat `key=value` dump of the resolved runtime config (vLLM-style).

        The resolved HF config objects are left out: they are huge and already
        summed up by the fields below. Values that the engine refines at load
        time (kvcache_block_size, num_kvcache_blocks, max_num_seqs, ...) reflect
        those refinements once logging happens.
        """
        return (
            f"model={self.model!r}, "
            f"architectures={getattr(self.text_config, 'architectures', None)}, "
            f"dtype={self.text_config.dtype}, "
            f"tensor_parallel_size={self.tensor_parallel_size}, "
            f"max_model_len={self.max_model_len}, "
            f"max_num_batched_tokens={self.max_num_batched_tokens}, "
            f"max_num_seqs={self.max_num_seqs}, "
            f"gpu_memory_utilization={self.gpu_memory_utilization}, "
            f"kvcache_block_size={self.kvcache_block_size}, "
            f"num_kvcache_blocks={self.num_kvcache_blocks}, "
            f"has_linear_attention={self.has_linear_attention}, "
            f"num_linear_state_groups={self.num_linear_state_groups}, "
            f"snapshot_decode_states={self.snapshot_decode_states}, "
            f"enable_mm_cache={self.enable_mm_cache}, "
            f"mm_cache_size_gb={self.mm_cache_size_gb}, "
            f"enforce_eager={self.enforce_eager}, "
            f"profile={self.profile}, "
            f"trust_remote_code={self.trust_remote_code}, "
            f"rdzv_port={self.rdzv_port}"
        )
