import importlib.metadata

from cs336_basics.data import get_batch, load_checkpoint, save_checkpoint
from cs336_basics.model import (
    CausalMultiHeadSelfAttention,
    Embedding,
    Linear,
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLU,
    TransformerBlock,
    TransformerLM,
    generate,
    scaled_dot_product_attention,
    silu,
    softmax,
)
from cs336_basics.optim import (
    AdamW,
    clip_gradients,
    cross_entropy,
    get_lr_cosine_schedule,
)
from cs336_basics.tokenizer import Tokenizer, train_bpe

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "AdamW",
    "CausalMultiHeadSelfAttention",
    "Embedding",
    "Linear",
    "RMSNorm",
    "RotaryPositionalEmbedding",
    "SwiGLU",
    "Tokenizer",
    "TransformerBlock",
    "TransformerLM",
    "clip_gradients",
    "cross_entropy",
    "generate",
    "get_batch",
    "get_lr_cosine_schedule",
    "load_checkpoint",
    "save_checkpoint",
    "scaled_dot_product_attention",
    "silu",
    "softmax",
    "train_bpe",
]
