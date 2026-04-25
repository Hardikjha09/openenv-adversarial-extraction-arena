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
- adversarial-robustness
- structured-extraction
- json-schema
---

# Extractor (SFT warmup) — Adversarial Structured-Extraction Arena

This model repo hosts the **SFT warmup LoRA adapter** trained for the OpenEnv project **Adversarial Structured-Extraction Arena**: an adversary perturbs messy documents/schemas (under a budget) and the extractor must output **valid JSON** matching a target schema.

## Links (submission)

- **GitHub repo**: https://github.com/Hardikjha09/openenv-adversarial-extraction-arena
- **Runnable Space**: https://huggingface.co/spaces/HardikJha/extraction-arena
- **Colab (re-run training)**: https://colab.research.google.com/github/Hardikjha09/openenv-adversarial-extraction-arena/blob/main/notebooks/Train_Extractor_Colab.ipynb
- 
## Evidence (plots + logs)

- **Training loss**: https://huggingface.co/HardikJha/extractor-aea/blob/main/plots/sft_loss.png
- **Eval reward (moving average)**: https://huggingface.co/HardikJha/extractor-aea/blob/main/plots/rewards.png
- **Eval Elo**: https://huggingface.co/HardikJha/extractor-aea/blob/main/plots/elo_ratings.png
- **Eval metrics JSON**: https://huggingface.co/HardikJha/extractor-aea/blob/main/eval_metrics.json
- **SFT trainer log (raw JSON)**: https://huggingface.co/HardikJha/extractor-aea/blob/main/trainer_log_history.json
- 
## What this checkpoint is

- **Base model**: `unsloth/Qwen2.5-1.5B-Instruct` (4-bit Unsloth bundle: `unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit`)
- **Adapter**: LoRA (`peft`), saved from `training/sft_warmup.py`
- 
## Quick start (load base + adapter)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "unsloth/Qwen2.5-1.5B-Instruct"
adapter_id = "HardikJha/extractor-aea"

tokenizer = AutoTokenizer.from_pretrained(adapter_id, trust_remote_code=True)

base = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base, adapter_id)
```

## Training procedure
- **Objective**:  supervised JSON extraction formatting aligned to the repo’s extractor prompt (`training/prompts.py`)
- **Framework**: TRL SFTTrainer + Unsloth FastLanguageModel (see `training/sft_warmup.py`)

This model was trained with SFT.

### Framework versions

- PEFT 0.18.1
- TRL: 0.23.0
- Transformers: 4.57.2
- Pytorch: 2.10.0+cu128
- Datasets: 4.3.0
- Tokenizers: 0.22.2

## Citations

Cite TRL as:
    
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