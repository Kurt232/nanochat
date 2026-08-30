"""Export a nanochat Qwen3 checkpoint as a Transformers model repository."""

import argparse
import base64
import json
import pickle
import shutil
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
from transformers.integrations.tiktoken import TikTokenConverter

from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    return parser.parse_args()


def export_tokenizer(source_dir, output_dir, model_max_length):
    with (source_dir / "tokenizer.pkl").open("rb") as handle:
        encoding = pickle.load(handle)

    # This is the on-disk tiktoken format: base64 token bytes followed by rank.
    vocab_file = output_dir / "tokenizer.model"
    with vocab_file.open("wb") as handle:
        for token, rank in sorted(
            encoding._mergeable_ranks.items(), key=lambda item: item[1]
        ):
            handle.write(
                base64.b64encode(token) + b" " + str(rank).encode() + b"\n"
            )

    backend = TikTokenConverter(
        vocab_file=str(vocab_file),
        pattern=encoding._pat_str,
        additional_special_tokens=encoding._special_tokens,
    ).converted()
    tokenizer_json = output_dir / "tokenizer.json"
    backend.save(str(tokenizer_json))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json),
        bos_token="<|bos|>",
        additional_special_tokens=SPECIAL_TOKENS[1:],
    )
    tokenizer.model_max_length = model_max_length
    tokenizer.save_pretrained(output_dir)

    native = RustBPETokenizer(encoding, "<|bos|>")
    samples = [
        "hello world",
        "你好，世界！",
        "Qwen3 nanochat 1234567890",
        "line one\nline two\t🙂",
        "",
    ]
    assert len(tokenizer) == native.get_vocab_size()
    for sample in samples:
        expected = native.encode(sample)
        actual = tokenizer.encode(sample, add_special_tokens=False)
        assert actual == expected, f"Tokenizer mismatch for {sample!r}"
    for token in SPECIAL_TOKENS:
        assert tokenizer.convert_tokens_to_ids(token) == native.encode_special(token)

    original_dir = output_dir / "nanochat_tokenizer"
    original_dir.mkdir()
    for filename in ("tokenizer.pkl", "token_bytes.pt"):
        shutil.copy2(source_dir / filename, original_dir / filename)
    return tokenizer


def build_hf_config(model_config, bos_token_id, dtype):
    return Qwen3Config(
        vocab_size=model_config["vocab_size"],
        hidden_size=model_config["hidden_size"],
        intermediate_size=model_config["intermediate_size"],
        num_hidden_layers=model_config["num_hidden_layers"],
        num_attention_heads=model_config["num_attention_heads"],
        num_key_value_heads=model_config["num_key_value_heads"],
        head_dim=model_config["head_dim"],
        hidden_act=model_config["hidden_act"],
        max_position_embeddings=model_config["max_position_embeddings"],
        initializer_range=model_config["initializer_range"],
        rms_norm_eps=model_config["rms_norm_eps"],
        rope_theta=model_config["rope_theta"],
        attention_bias=model_config["attention_bias"],
        attention_dropout=model_config["attention_dropout"],
        tie_word_embeddings=model_config["tie_word_embeddings"],
        bos_token_id=bos_token_id,
        eos_token_id=None,
        pad_token_id=None,
        use_cache=True,
        dtype=dtype,
    )


def export_model(checkpoint_path, config, dtype, output_dir):
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = {
        key.removeprefix("_orig_mod."): value.to(dtype=dtype)
        if value.is_floating_point()
        else value
        for key, value in state_dict.items()
    }
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    incompatible = model.load_state_dict(state_dict, strict=True, assign=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    model.tie_weights()
    model.eval()
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    return sum(parameter.numel() for parameter in model.parameters())


def write_model_card(output_dir, meta, num_parameters, dtype):
    model_config = meta["model_config"]
    total_tokens = meta["step"] * meta["total_batch_size"]
    text = f"""---
pipeline_tag: text-generation
library_name: transformers
tags:
- qwen3
- nanochat
- base-model
---

# Nanochat Qwen3 1B Base

This is a base language model trained from scratch with nanochat and the Muon
optimizer. It is not instruction-tuned or chat-tuned.

## Architecture and training

- Parameters: {num_parameters:,}
- Hidden size: {model_config['hidden_size']}
- Intermediate size: {model_config['intermediate_size']}
- Layers: {model_config['num_hidden_layers']}
- Attention heads: {model_config['num_attention_heads']} query / {model_config['num_key_value_heads']} KV
- Head dimension: {model_config['head_dim']}
- Vocabulary: {model_config['vocab_size']:,}, trained by nanochat
- Context used for pretraining: {model_config['sequence_len']:,}
- Training tokens: {total_tokens:,}
- Optimizer: MuonAdamW
- Published weight dtype: {str(dtype).removeprefix('torch.')}
- Final validation BPB: {meta['val_bpb']:.6f}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("REPO_ID")
model = AutoModelForCausalLM.from_pretrained(
    "REPO_ID", dtype="auto", device_map="auto"
)
inputs = tokenizer("Once upon a time", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(output[0]))
```

The original nanochat tokenizer artifacts are retained in
`nanochat_tokenizer/`. The full Muon optimizer checkpoint is retained on HDFS,
not in this inference repository.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    model_config = meta["model_config"]
    dtype = DTYPES[args.dtype]
    tokenizer = export_tokenizer(
        args.tokenizer_dir,
        args.output_dir,
        model_config["max_position_embeddings"],
    )
    config = build_hf_config(model_config, tokenizer.bos_token_id, dtype)
    num_parameters = export_model(args.checkpoint, config, dtype, args.output_dir)
    shutil.copy2(args.meta, args.output_dir / "training_meta.json")
    write_model_card(args.output_dir, meta, num_parameters, dtype)
    print(f"Exported {num_parameters:,} parameters to {args.output_dir}")


if __name__ == "__main__":
    main()
