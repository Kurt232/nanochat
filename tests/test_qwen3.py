import pytest
import torch

from nanochat.qwen3 import Qwen3, Qwen3Config


def tiny_config():
    return Qwen3Config(
        sequence_len=32,
        vocab_size=256,
        hidden_size=64,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=32,
    )


def build_model(config):
    with torch.device("meta"):
        model = Qwen3(config, logits_chunk_size=8)
    model.to_empty(device="cpu")
    model.init_weights()
    return model


class TinyKVCache:
    def __init__(self, config, batch_size, seq_len):
        shape = (
            config.num_hidden_layers,
            batch_size,
            seq_len,
            config.num_key_value_heads,
            config.head_dim,
        )
        self.k_cache = torch.zeros(shape)
        self.v_cache = torch.zeros(shape)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32)
        self.n_layers = config.num_hidden_layers

    def get_layer_cache(self, layer_idx):
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def get_pos(self):
        return self.cache_seqlens[0].item()

    def advance(self, count):
        self.cache_seqlens += count


def test_qwen3_06b_topology():
    config = Qwen3Config.from_variant("qwen3-0.6b")
    assert config.num_hidden_layers == 28
    assert config.hidden_size == 1024
    assert config.intermediate_size == 3072
    assert config.num_attention_heads == 16
    assert config.num_key_value_heads == 8
    assert config.head_dim == 128
    assert config.rope_theta == 1_000_000
    assert config.max_position_embeddings == 40_960
    assert config.tie_word_embeddings

    # nanochat's 32K tokenizer makes this smaller than the official checkpoint,
    # whose tied embedding uses a 151,936-token vocabulary.
    with torch.device("meta"):
        model = Qwen3(config)
    assert sum(p.numel() for p in model.parameters()) == 474_021_888
    for name, param in model.named_parameters():
        if param.ndim != 2 or name == "model.embed_tokens.weight":
            assert param.shape[0] % 32 == 0, f"{name} cannot be ZeRO-2 sharded over 32 ranks"


@pytest.mark.parametrize(
    "variant,expected",
    [
        ("qwen3-scale-48m", (8, 512, 1536, 8, 4, 48_245_248)),
        ("qwen3-scale-131m", (12, 768, 2304, 12, 6, 131_356_416)),
        ("qwen3-scale-285m", (16, 1024, 3072, 16, 8, 285_250_560)),
    ],
)
def test_qwen3_scaling_topologies(variant, expected):
    config = Qwen3Config.from_variant(variant)
    layers, hidden, intermediate, heads, kv_heads, parameters = expected
    assert config.num_hidden_layers == layers
    assert config.hidden_size == hidden
    assert config.intermediate_size == intermediate
    assert config.num_attention_heads == heads
    assert config.num_key_value_heads == kv_heads
    with torch.device("meta"):
        model = Qwen3(config)
    assert sum(p.numel() for p in model.parameters()) == parameters


def test_qwen3_1b_topology():
    config = Qwen3Config.from_variant("qwen3-1b")
    assert config.num_hidden_layers == 28
    assert config.hidden_size == 1536
    assert config.intermediate_size == 5376
    assert config.num_attention_heads == 16
    assert config.num_key_value_heads == 8
    assert config.head_dim == 128
    with torch.device("meta"):
        model = Qwen3(config)
    assert sum(p.numel() for p in model.parameters()) == 1_008_300_544
    assert model.estimate_flops() == 8_867_807_232


def test_qwen3_forward_backward_and_tied_embedding():
    model = build_model(tiny_config())
    assert model.lm_head.weight is model.model.embed_tokens.weight
    optimizer = model.setup_optimizer(embedding_lr=3e-4, scalar_lr=3e-4)
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    assert len({id(p) for p in optimized}) == len(list(model.parameters()))
    assert sum(p is model.model.embed_tokens.weight for p in optimized) == 1
    ids = torch.randint(0, model.config.vocab_size, (2, 12))
    targets = torch.randint(0, model.config.vocab_size, (2, 12))
    loss = model(ids, targets)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert model.model.embed_tokens.weight.grad is not None


def test_qwen3_kv_cache_matches_full_forward():
    model = build_model(tiny_config()).eval()
    ids = torch.randint(0, model.config.vocab_size, (2, 6))
    cache = TinyKVCache(model.config, batch_size=2, seq_len=16)
    with torch.no_grad():
        prefill = model(ids[:, :5], kv_cache=cache)
        decode = model(ids[:, 5:], kv_cache=cache)
        full5 = model(ids[:, :5])
        full6 = model(ids)
    torch.testing.assert_close(prefill[:, -1], full5[:, -1])
    torch.testing.assert_close(decode[:, -1], full6[:, -1], rtol=1e-5, atol=1e-6)
    assert cache.get_pos() == 6


def test_forward_matches_transformers_qwen3():
    transformers = pytest.importorskip("transformers")
    from transformers import Qwen3Config as HFQwen3Config
    from transformers import Qwen3ForCausalLM

    config = tiny_config()
    native = build_model(config).eval()
    hf_config = HFQwen3Config(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_position_embeddings=config.sequence_len,
        hidden_act=config.hidden_act,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        attention_bias=config.attention_bias,
        attention_dropout=config.attention_dropout,
        tie_word_embeddings=True,
        use_cache=False,
    )
    hf_config._attn_implementation = "sdpa"
    hf = Qwen3ForCausalLM(hf_config).eval()
    hf.load_state_dict(native.state_dict(), strict=True)

    ids = torch.randint(0, config.vocab_size, (2, 12))
    with torch.no_grad():
        actual = native(ids)
        expected = hf(ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
