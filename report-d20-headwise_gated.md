# nanochat training report

Generated: 2025-12-20 21:25:59

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
- Characters: 422,206
- Lines: 10,244
- Files: 49
- Tokens (approx): 105,551
- Dependencies (uv.lock lines): 2,218

Run started: 2025-12-20 21:25:59

---

## Tokenizer evaluation
timestamp: 2025-12-20 21:26:08

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
timestamp: 2025-12-20 21:27:14

- train bpb: 0.8179
- val bpb: 0.8131
- sample 0: <|bos|>The capital of France is Paris, and the capital of France is Paris. The capital of France is Paris
- sample 1: <|bos|>The chemical symbol of gold is Au. The chemical symbol of silver is Ag. The chemical symbol of copper is
- sample 2: <|bos|>If yesterday was Friday, then tomorrow will be Monday. The day before that, the day after that, the day after that
- sample 3: <|bos|>The opposite of hot is cold. The opposite of hot is cold. The opposite of hot is cold.
- sample 4: <|bos|>The planets of the solar system are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune,
- sample 5: <|bos|>My favorite color is red. I love it because it is so bright and vibrant. I love it
- sample 6: <|bos|>If 5*x + 3 = 13, then x is a prime number. What is the greatest prime number?
The greatest prime number is


## Base model evaluation
timestamp: 2025-12-20 21:38:38

- Model: base_model (step 21400)
- CORE metric: 0.2172
- hellaswag_zeroshot: 0.2628
- jeopardy: 0.0827
- bigbench_qa_wikidata: 0.5088
- arc_easy: 0.5325
- arc_challenge: 0.1058
- copa: 0.3400
- commonsense_qa: 0.1360
- piqa: 0.3569
- openbook_qa: 0.1173
- lambada_openai: 0.3792
- hellaswag: 0.2602
- winograd: 0.2967
- winogrande: 0.0797
- bigbench_dyck_languages: 0.1770
- agi_eval_lsat_ar: 0.0815
- bigbench_cs_algorithms: 0.3902
- bigbench_operators: 0.1429
- bigbench_repeat_copy_logic: 0.0312
- squad: 0.2623
- coqa: 0.1993
- boolq: -0.1395
- bigbench_language_identification: 0.1752


## Chat evaluation mid
timestamp: 2025-12-20 22:00:26

- source: mid
- task_name: None
- dtype: bfloat16
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: d20-headwise_gated
- step: None
- max_problems: None
- device_type: 
- ARC-Easy: 0.4301
- ARC-Challenge: 0.3225
- MMLU: 0.3213
- GSM8K: 0.0417
- HumanEval: 0.0366
- SpellingBee: 0.9766
- ChatCORE metric: 0.2478


## Chat evaluation sft
timestamp: 2025-12-20 22:17:18

- source: sft
- task_name: None
- dtype: bfloat16
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: d20-headwise_gated
- step: None
- max_problems: None
- device_type: 
- ARC-Easy: 0.4592
- ARC-Challenge: 0.3336
- MMLU: 0.3291
- GSM8K: 0.0607
- HumanEval: 0.0427
- SpellingBee: 0.9844
- ChatCORE metric: 0.2639


## Summary

- Characters: 422,206
- Lines: 10,244
- Files: 49
- Tokens (approx): 105,551
- Dependencies (uv.lock lines): 2,218

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.2172   | -        | -        | -        |
| ARC-Challenge   | -        | 0.3225   | 0.3336   | -        |
| ARC-Easy        | -        | 0.4301   | 0.4592   | -        |
| GSM8K           | -        | 0.0417   | 0.0607   | -        |
| HumanEval       | -        | 0.0366   | 0.0427   | -        |
| MMLU            | -        | 0.3213   | 0.3291   | -        |
| ChatCORE        | -        | 0.2478   | 0.2639   | -        |

Total wall clock time: 0h51m
