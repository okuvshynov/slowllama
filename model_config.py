import torch

from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0 # unless we bring back 
    ffn_dim_multiplier: Optional[float] = None
    compute_dtype: torch.dtype = torch.float32
    rope_theta: float = 10000.0
    lora_rank: int = 8
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    served_model_path: str = '' # relative path by default
    cached_data_path: str = ''  # relative path by default
    init_frozen: bool = True
    frozen_dtype: torch.dtype = torch.bfloat16
    vocab_size: int = 32000
    vocab_size_override: int = 32000