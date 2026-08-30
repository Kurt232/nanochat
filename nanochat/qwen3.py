"""Nanochat-native Qwen3 model.

The module topology and forward math mirror Transformers' Qwen3ForCausalLM,
while the training/inference interfaces remain native to nanochat.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE, print0
from nanochat.flash_attention import flash_attn
from nanochat.gpt import Linear
from nanochat.optim import MuonAdamW


@dataclass
class Qwen3Config:
    architecture: str = "qwen3"
    sequence_len: int = 4096
    vocab_size: int = 32768
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 40_960
    attention_bias: bool = False
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True

    # Compatibility aliases used by nanochat's Engine and training utilities.
    @property
    def n_layer(self):
        return self.num_hidden_layers

    @property
    def n_head(self):
        return self.num_attention_heads

    @property
    def n_kv_head(self):
        return self.num_key_value_heads

    @property
    def n_embd(self):
        return self.hidden_size

    @classmethod
    def from_variant(cls, variant, *, sequence_len=4096, vocab_size=32768):
        """Return one of the hard-coded Qwen3-family topologies used by nanochat."""
        variants = {
            # Small Qwen3 miniseries points used for scaling-law measurements.
            # They retain Qwen3's 128-dim QK-normalized heads and 2:1 GQA while
            # scaling depth and width together, in the style of nanochat.
            "qwen3-scale-48m": {
                "hidden_size": 512,
                "intermediate_size": 1536,
                "num_hidden_layers": 8,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
            },
            "qwen3-scale-131m": {
                "hidden_size": 768,
                "intermediate_size": 2304,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "num_key_value_heads": 6,
            },
            "qwen3-scale-285m": {
                "hidden_size": 1024,
                "intermediate_size": 3072,
                "num_hidden_layers": 16,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
            },
            "qwen3-0.6b": {},
            # A Qwen3-style ~1B model for nanochat's 32K tied vocabulary.
            # Keep the 0.6B/1.7B family's 28 layers, 16Q/8KV GQA and 128 head dim.
            "qwen3-1b": {
                "hidden_size": 1536,
                "intermediate_size": 5376,
            },
        }
        if variant not in variants:
            raise ValueError(f"Unknown Qwen3 variant: {variant}")
        return cls(sequence_len=sequence_len, vocab_size=vocab_size, **variants[variant])


class Qwen3RMSNorm(nn.Module):
    """Qwen3 RMSNorm, including its learnable scale."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.square().mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.to(input_dtype) * hidden_states.to(input_dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Qwen3Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.q_proj = Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, cos_sin, kv_cache=None):
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # Transformers Qwen3 applies per-head QK norm before RoPE.
        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = cos_sin
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, 0))
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                k=k,
                v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=(-1, 0),
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(seq_len)

        return self.o_proj(y.contiguous().view(batch_size, seq_len, -1))


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, cos_sin, kv_cache=None):
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states), cos_sin, kv_cache)
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3(nn.Module):
    """Qwen3ForCausalLM topology with nanochat's model API and MuonAdamW."""

    def __init__(self, config, logits_chunk_size=1024):
        super().__init__()
        assert config.hidden_act == "silu"
        assert config.tie_word_embeddings, "Qwen3 uses tied token embeddings"
        assert not config.attention_bias, "Qwen3 attention projections are bias-free"
        assert config.attention_dropout == 0.0, "Qwen3 uses zero attention dropout"
        assert config.num_attention_heads % config.num_key_value_heads == 0
        assert config.head_dim % 2 == 0
        self.config = config
        self.logits_chunk_size = logits_chunk_size
        self.model = Qwen3Model(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_weights()

        self.rotary_seq_len = max(config.sequence_len, config.max_position_embeddings)
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Match Transformers Qwen3 initialization. Tied lm_head shares the embedding tensor.
        for module in self.modules():
            if isinstance(module, Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, Qwen3RMSNorm):
                nn.init.ones_(module.weight)
        self.tie_weights()
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len)
        self.cos, self.sin = cos, sin
        if COMPUTE_DTYPE != torch.float16:
            self.model.embed_tokens.to(dtype=COMPUTE_DTYPE)

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def _precompute_rotary_embeddings(self, seq_len, device=None):
        if device is None:
            device = self.model.embed_tokens.weight.device
        inv_freq = 1.0 / (
            self.config.rope_theta
            ** (torch.arange(0, self.config.head_dim, 2, dtype=torch.float32, device=device) / self.config.head_dim)
        )
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(COMPUTE_DTYPE)[None, :, None, :]
        sin = emb.sin().to(COMPUTE_DTYPE)[None, :, None, :]
        return cos, sin

    def get_device(self):
        return self.model.embed_tokens.weight.device

    def num_matmul_params(self):
        # lm_head is intentionally counted: tied weights still perform an output matmul.
        return sum(module.weight.numel() for module in self.modules() if isinstance(module, Linear))

    def estimate_flops(self):
        attention_flops = (
            12
            * self.config.num_hidden_layers
            * self.config.num_attention_heads
            * self.config.head_dim
            * self.config.sequence_len
        )
        return 6 * self.num_matmul_params() + attention_flops

    def estimate_decode_flops(self, context_len):
        attn = 4 * self.config.num_hidden_layers * self.config.num_attention_heads * self.config.head_dim * context_len
        return 2 * self.num_matmul_params() + attn

    def estimate_prefill_flops(self, num_tokens):
        attended = num_tokens * (num_tokens + 1) // 2
        attn = 4 * self.config.num_hidden_layers * self.config.num_attention_heads * self.config.head_dim * attended
        return 2 * self.num_matmul_params() * num_tokens + attn

    def kv_bytes_per_token(self):
        return (
            self.config.num_hidden_layers
            * 2
            * self.config.num_key_value_heads
            * self.config.head_dim
            * COMPUTE_DTYPE.itemsize
        )

    def kv_read_bytes(self, context_len):
        return self.kv_bytes_per_token() * context_len

    def num_scaling_params(self):
        embedding = self.model.embed_tokens.weight.numel()
        transformer = sum(p.numel() for p in self.model.layers.parameters()) + self.model.norm.weight.numel()
        total = sum(p.numel() for p in self.parameters())
        assert total == embedding + transformer
        return {
            "wte": embedding,
            "value_embeds": 0,
            # Logical output projection size. It overlaps wte because Qwen3 ties the weights.
            "lm_head": embedding,
            "transformer_matrices": transformer,
            "scalars": 0,
            "total": total,
        }

    def setup_optimizer(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        scalar_lr=0.0003,
    ):
        del unembedding_lr  # lm_head is tied to the embedding parameter
        embedding_param = self.model.embed_tokens.weight
        matrix_params = []
        norm_params = []
        for name, param in self.named_parameters():
            if param is embedding_param:
                continue
            if param.ndim == 2:
                matrix_params.append(param)
            else:
                norm_params.append(param)

        covered = {id(embedding_param), *(id(p) for p in matrix_params), *(id(p) for p in norm_params)}
        assert covered == {id(p) for p in self.parameters()}, "Optimizer parameter grouping mismatch"

        dmodel_lr_scale = (self.config.hidden_size / 768) ** -0.5
        print0(f"Scaling AdamW LRs ∝1/√dmodel: {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(
                kind="adamw",
                params=[embedding_param],
                lr=embedding_lr * dmodel_lr_scale,
                betas=(0.8, 0.995),
                eps=1e-10,
                weight_decay=0.001,
            ),
            dict(
                kind="adamw",
                params=norm_params,
                lr=scalar_lr * dmodel_lr_scale,
                betas=(0.9, 0.95),
                eps=1e-10,
                weight_decay=0.0,
            ),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            param_groups.append(
                dict(
                    kind="muon",
                    params=[p for p in matrix_params if p.shape == shape],
                    lr=matrix_lr,
                    momentum=0.95,
                    ns_steps=5,
                    beta2=0.9,
                    weight_decay=weight_decay,
                )
            )
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def _loss(self, hidden_states, targets, reduction):
        flat_hidden = hidden_states.reshape(-1, hidden_states.size(-1))
        flat_targets = targets.reshape(-1)
        losses = []
        for start in range(0, flat_hidden.size(0), self.logits_chunk_size):
            end = min(start + self.logits_chunk_size, flat_hidden.size(0))
            logits = self.lm_head(flat_hidden[start:end]).float()
            losses.append(F.cross_entropy(logits, flat_targets[start:end], ignore_index=-1, reduction="none"))
        loss = torch.cat(losses)
        if reduction == "none":
            return loss.view_as(targets)
        if reduction == "sum":
            return loss.sum()
        valid = (flat_targets != -1).sum().clamp_min(1)
        return loss.sum() / valid

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        _, seq_len = idx.shape
        start = 0 if kv_cache is None else kv_cache.get_pos()
        assert start + seq_len <= self.cos.size(1)
        cos_sin = self.cos[:, start : start + seq_len], self.sin[:, start : start + seq_len]

        hidden_states = self.model.embed_tokens(idx).to(COMPUTE_DTYPE)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states, cos_sin, kv_cache)
        hidden_states = self.model.norm(hidden_states)

        if targets is not None:
            return self._loss(hidden_states, targets, loss_reduction)
        return self.lm_head(hidden_states).float()

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        device = self.get_device()
        rng = torch.Generator(device=device).manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self(ids)[:, -1]
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, 1, generator=rng)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            token = next_id.item()
            yield token
            ids = torch.cat((ids, next_id), dim=1)
