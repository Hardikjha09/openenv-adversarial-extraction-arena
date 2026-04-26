---
base_model: unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit
library_name: peft
language:
- en
license: apache-2.0
pipeline_tag: text-generation
tags:
- base_model:adapter:unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit
- lora
- sft
- transformers
- trl
- unsloth
- openenv
- adversarial-ml
- structured-extraction
---

# Adversary policy (SFT LoRA) — Adversarial Structured-Extraction Arena

This repo hosts the **SFT warmup LoRA adapter** for the **Adversary** agent in [Adversarial Structured-Extraction Arena](https://github.com/Hardikjha09/openenv-adversarial-extraction-arena): given a document, a JSON schema, and a perturbation budget, the model proposes a **JSON list of edits** (`rename_field`, `ocr_noise`, `swap_type`, `inject_distractor`, …). The environment applies those edits via `env/adversary.py`; the paired extractor must still emit valid JSON for the target schema.

## Links (submission)

- **GitHub repo**: https://github.com/Hardikjha09/openenv-adversarial-extraction-arena
- **Runnable Space**: https://huggingface.co/spaces/HardikJha/extraction-arena
- **Colab (train extractor + adversary)**: https://colab.research.google.com/github/Hardikjha09/openenv-adversarial-extraction-arena/blob/main/notebooks/Train_Extractor_Colab.ipynb
- **Paired extractor LoRA**: https://huggingface.co/HardikJha/extractor-aea

## Evidence (plots + logs)

- **Training loss**: https://huggingface.co/HardikJha/adversary-aea/blob/main/plots/sft_adversary_loss.png
- **SFT trainer log (raw JSON)**: https://huggingface.co/HardikJha/adversary-aea/blob/main/trainer_log_history.json

## What this checkpoint is

- **Base model**: `unsloth/Qwen2.5-1.5B-Instruct` (4-bit Unsloth bundle: `unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit`)
- **Adapter**: LoRA (`peft`), saved from `training/sft_adversary.py`
- **Supervision**: **Heuristic** edit programs sampled per document from `data/corpus.json` (same edit types the executor supports). Default training slice: **200 documents** starting at index **200** (so default SFT does not overlap the extractor’s first-200-doc slice).
- **I/O**: User message is `ADVERSARY_SYSTEM_PROMPT` in `training/prompts.py` (document + schema + budget). Target completion is a single fenced JSON array of edit objects.

## Quick start (load base + adapter)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "unsloth/Qwen2.5-1.5B-Instruct"
adapter_id = "HardikJha/adversary-aea"

tokenizer = AutoTokenizer.from_pretrained(adapter_id, trust_remote_code=True)

base = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base, adapter_id)
```

For a minimal “generate edits” call, build the prompt with `ADVERSARY_SYSTEM_PROMPT.format(budget=..., document=..., schema=...)` from the repo, then decode the model output and parse the JSON inside the model’s fenced code block (see `evaluation/run_eval.py`).

## Train (reproduce)

```bash
PYTHONPATH=. python training/sft_adversary.py \
  --model_name unsloth/Qwen2.5-1.5B-Instruct \
  --output_dir checkpoints/sft_adversary \
  --corpus_path data/corpus.json \
  --start_idx 200 \
  --n_docs 200
```

Optional: log history is written to `checkpoints/sft_adversary/trainer_log_history.json`. Plot with `python plots/generate_training_plots.py` (includes `sft_adversary` loss).

## Eval (trained extractor + trained adversary)

```bash
PYTHONPATH=. python evaluation/run_eval.py \
  --model_path checkpoints/sft_warmup \
  --adversary_model_path checkpoints/sft_adversary \
  --num_episodes 50
```

Use local checkpoint paths, or point `--adversary_model_path` at a directory that contains the saved adapter (same layout as after `sft_adversary.py`).

## Training procedure

- **Objective**: Supervised next-token prediction on concatenated prompt + target edit JSON (fenced), aligned to the adversary prompt in `training/prompts.py`.
- **Framework**: TRL `SFTTrainer` + Unsloth `FastLanguageModel` (see `training/sft_adversary.py`).
- **LoRA**: `r=16`, `lora_alpha=16`, target modules `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`; 4-bit base; `max_seq_length=2048`; default run `max_steps=50`, batch 2, grad accum 4, LR `2e-4`.

### Framework versions (reference; match your env via `pip freeze`)

Versions below match a typical Colab/Unsloth install; pin to your run for reproducibility.

- PEFT 0.18.1
- TRL 0.23.0
- Transformers 4.57.2
- PyTorch 2.10.0+cu128
- Datasets 4.3.0
- Tokenizers 0.22.2

## Limitations

- Labels are **synthetic heuristics**, not human or RL-optimal attacks; the policy is a **warm start** for arena evaluation, not a guaranteed strongest adversary.
- Out-of-distribution documents or schemas may yield invalid JSON or edit types not accepted by the executor—evaluation code falls back when parsing fails.

## Citations

```bibtex
@misc{vonwerra2022trl,
  title        = {{TRL: Transformer Reinforcement Learning}},
  author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
  year         = 2020,
  journal      = {GitHub repository},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/huggingface/trl}}
}
```

Also cite **Qwen2.5** and **Unsloth** per their model cards if you use this checkpoint in a paper or report.
