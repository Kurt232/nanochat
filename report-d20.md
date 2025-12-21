# nanochat training report

Generated: 2025-12-20 22:30:20

## Environment

### Git Information
- Branch: master
- Commit: 065d3a3 (dirty)
- Message: feature gated attention

### Hardware
- Platform: Linux
- CPUs: 128 cores (256 logical)
- Memory: 251.7 GB
- GPUs: 4x NVIDIA A800 80GB PCIe
- GPU Memory: 317.0 GB total
- CUDA Version: 12.8
- Hourly Rate: $8.00/hour

### Software
- Python: 3.10.12
- PyTorch: 2.8.0+cu128


### Bloat
- Characters: 427,307
- Lines: 10,425
- Files: 50
- Tokens (approx): 106,826
- Dependencies (uv.lock lines): 2,218

Run started: 2025-12-20 22:30:20

---

## Tokenizer evaluation
timestamp: 2025-12-20 22:30:30

### Comparison with GPT-2

| Text Type | Bytes | GPT-2 Tokens | GPT-2 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 404 | 4.50 | 375 | 4.85 | +7.2% |
| korean | 893 | 745 | 1.20 | 721 | 1.24 | +3.2% |
| code | 1259 | 576 | 2.19 | 493 | 2.55 | +14.4% |
| math | 1834 | 936 | 1.96 | 966 | 1.90 | -3.2% |
| science | 1112 | 260 | 4.28 | 225 | 4.94 | +13.5% |
| fwe-train | 4208518 | 900364 | 4.67 | 856901 | 4.91 | +4.8% |
| fwe-val | 4908443 | 1059062 | 4.63 | 1010356 | 4.86 | +4.6% |

### Comparison with GPT-4

| Text Type | Bytes | GPT-4 Tokens | GPT-4 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 387 | 4.70 | 375 | 4.85 | +3.1% |
| korean | 893 | 364 | 2.45 | 721 | 1.24 | -98.1% |
| code | 1259 | 309 | 4.07 | 493 | 2.55 | -59.5% |
| math | 1834 | 832 | 2.20 | 966 | 1.90 | -16.1% |
| science | 1112 | 249 | 4.47 | 225 | 4.94 | +9.6% |
| fwe-train | 4208518 | 874799 | 4.81 | 856901 | 4.91 | +2.0% |
| fwe-val | 4908443 | 1029691 | 4.77 | 1010356 | 4.86 | +1.9% |


## Base model loss
timestamp: 2025-12-20 22:31:34

- train bpb: 0.8158
- val bpb: 0.8111
- sample 0: <|bos|>The capital of France is Paris, and the capital of the European Union is Brussels. The capital of the
- sample 1: <|bos|>The chemical symbol of gold is Au. It is a soft, silvery-white metal that is malleable and ductile.
- sample 2: <|bos|>If yesterday was Friday, then tomorrow will be Monday. If today is Tuesday, then tomorrow will be Wednesday. If tomorrow is
- sample 3: <|bos|>The opposite of hot is cold. The opposite of cold is hot. The opposite of hot is cold.
- sample 4: <|bos|>The planets of the solar system are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune,
- sample 5: <|bos|>My favorite color is blue. I love it. I love it. I love it. I love
- sample 6: <|bos|>If 5*x + 3 = 13, then x is
The answer is 13.
The answer is 13.
The answer is


## Base model evaluation
timestamp: 2025-12-20 22:42:28

- Model: base_model (step 21400)
- CORE metric: 0.2047
- hellaswag_zeroshot: 0.2620
- jeopardy: 0.0883
- bigbench_qa_wikidata: 0.5301
- arc_easy: 0.5303
- arc_challenge: 0.1217
- copa: 0.4200
- commonsense_qa: 0.1063
- piqa: 0.3874
- openbook_qa: 0.1387
- lambada_openai: 0.3854
- hellaswag: 0.2652
- winograd: 0.2381
- winogrande: 0.0339
- bigbench_dyck_languages: 0.1070
- agi_eval_lsat_ar: 0.0326
- bigbench_cs_algorithms: 0.3894
- bigbench_operators: 0.1810
- bigbench_repeat_copy_logic: 0.0000
- squad: 0.2434
- coqa: 0.2116
- boolq: -0.3536
- bigbench_language_identification: 0.1838


## Midtraining
timestamp: 2025-12-21 00:02:24

- run: d20
- device_type: 
- dtype: bfloat16
- num_iterations: -1
- max_seq_len: 2048
- device_batch_size: 32
- unembedding_lr: 0.0040
- embedding_lr: 0.2000
- matrix_lr: 0.0200
- init_lr_frac: 1.0000
- weight_decay: 0.0000
- eval_every: 150
- eval_tokens: 10,485,760
- total_batch_size: 524,288
- dry_run: 0
- Number of iterations: 811
- DDP world size: 4
- Minimum validation bpb: 0.3951


## Chat evaluation mid
timestamp: 2025-12-21 00:19:35

- source: mid
- task_name: None
- dtype: bfloat16
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: d20
- step: None
- max_problems: None
- device_type: 
- ARC-Easy: 0.4491
- ARC-Challenge: 0.3131
- MMLU: 0.3351
- GSM8K: 0.0341
- HumanEval: 0.0061
- SpellingBee: 0.9805
- ChatCORE metric: 0.2473


## Chat SFT
timestamp: 2025-12-21 01:03:25

- run: d20
- source: mid
- device_type: 
- dtype: bfloat16
- device_batch_size: 4
- num_epochs: 1
- num_iterations: -1
- target_examples_per_step: 32
- unembedding_lr: 0.0040
- embedding_lr: 0.2000
- matrix_lr: 0.0200
- weight_decay: 0.0000
- init_lr_frac: 0.0200
- eval_every: 100
- eval_steps: 100
- eval_metrics_every: 200
- eval_metrics_max_problems: 1024
- Training rows: 22,439
- Number of iterations: 701
- Training loss: 1.2486
- Validation loss: 0.9960


## Chat evaluation sft
timestamp: 2025-12-21 01:19:52

- source: sft
- task_name: None
- dtype: bfloat16
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: d20
- step: None
- max_problems: None
- device_type: 
- ARC-Easy: 0.4676
- ARC-Challenge: 0.3225
- MMLU: 0.3328
- GSM8K: 0.0508
- HumanEval: 0.0122
- SpellingBee: 0.9805
- ChatCORE metric: 0.2568


## Summary

- Characters: 427,307
- Lines: 10,425
- Files: 50
- Tokens (approx): 106,826
- Dependencies (uv.lock lines): 2,218

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.2047   | -        | -        | -        |
| ARC-Challenge   | -        | 0.3131   | 0.3225   | -        |
| ARC-Easy        | -        | 0.4491   | 0.4676   | -        |
| GSM8K           | -        | 0.0341   | 0.0508   | -        |
| HumanEval       | -        | 0.0061   | 0.0122   | -        |
| MMLU            | -        | 0.3351   | 0.3328   | -        |
| ChatCORE        | -        | 0.2473   | 0.2568   | -        |

Total wall clock time: 2h49m
