from __future__ import annotations
import math
import torch
import torch.nn as nn
from einops import einsum, rearrange


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        std = math.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x_fp32 * x_fp32, dim=-1, keepdim=True) + self.eps)
        normed = (x_fp32 / rms) * self.weight.to(torch.float32)
        return normed.to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = max(64, int(round((8.0 * d_model / 3.0) / 64.0)) * 64)
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        exponents = torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k
        freqs = 1.0 / (theta ** exponents)
        angles = torch.outer(positions, freqs)
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        x_even = x_fp32[..., 0::2]
        x_odd = x_fp32[..., 1::2]

        while cos.ndim < x_even.ndim and cos.ndim >= 3:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        out = torch.stack([out_even, out_odd], dim=-1).flatten(-2)
        return out.to(in_dtype)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... q d_k, ... k d_k -> ... q k") / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask.bool(), float("-inf"))
    probs = softmax(scores, dim=-1)
    return einsum(probs, V, "... q k, ... k d_v -> ... q d_v")


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = rope
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = x.shape[-2]
        w_qkv = torch.cat(
            [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0
        )
        qkv = einsum(x, w_qkv, "... s d_in, d_out d_in -> ... s d_out")
        q, k, v = qkv.chunk(3, dim=-1)

        q = rearrange(q, "... s (h d_k) -> ... h s d_k", h=self.num_heads)
        k = rearrange(k, "... s (h d_k) -> ... h s d_k", h=self.num_heads)
        v = rearrange(v, "... s (h d_k) -> ... h s d_k", h=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        causal_mask = torch.tril(
            torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool)
        )
        attn_out = scaled_dot_product_attention(q, k, v, mask=causal_mask)
        attn_out = rearrange(attn_out, "... h s d_k -> ... s (h d_k)")
        return self.output_proj(attn_out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if rope is None:
            rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=d_model // num_heads,
                max_seq_len=max_seq_len,
                device=device,
            )
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope=rope,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        shared_rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_k=d_model // num_heads,
            max_seq_len=context_length,
            device=device,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    rope=shared_rope,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        in_indices: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.token_embeddings(in_indices)
        if token_positions is None:
            seq_len = in_indices.shape[-1]
            token_positions = torch.arange(seq_len, device=in_indices.device)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        x = self.ln_final(x)
        return self.lm_head(x)


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_ids: list[int] | torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str | torch.device = "cpu",
) -> list[int]:
    model.eval()
    if isinstance(prompt_ids, list):
        seq = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    else:
        seq = prompt_ids.to(device=device, dtype=torch.long)
        if seq.ndim == 1:
            seq = seq.unsqueeze(0)

    for _ in range(max_new_tokens):
        idx_cond = seq[:, -model.context_length :]
        logits = model(idx_cond)[:, -1, :]

        if temperature <= 0.0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            scaled_logits = logits / temperature
            probs = softmax(scaled_logits, dim=-1)

            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative_probs - sorted_probs >= top_p
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices.gather(-1, sampled_idx)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

        seq = torch.cat([seq, next_token], dim=-1)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return seq.squeeze(0).tolist()
