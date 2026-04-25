---
license: apache-2.0
base_model: unsloth/Qwen2.5-1.5B-Instruct
tags:
  - openenv
  - adversarial-ml
  - lora
  - unsloth
  - structured-extraction
pipeline_tag: text-generation
---

# Adversary policy (LoRA) — `adversary-aea`

SFT warmup adapter for the **Adversary** agent in [Adversarial Structured-Extraction Arena](https://github.com/Hardikjha09/openenv-adversarial-extraction-arena): given a document + JSON schema + perturbation budget, the model emits a **JSON list of edits** (`rename_field`, `ocr_noise`, `swap_type`, `inject_distractor`, …) executed by `env/adversary.py`.

## Links

- **GitHub**: https://github.com/Hardikjha09/openenv-adversarial-extraction-arena
- **Paired extractor adapter**: https://huggingface.co/HardikJha/extractor-aea
- **Colab (train both)**: https://colab.research.google.com/github/Hardikjha09/openenv-adversarial-extraction-arena/blob/main/notebooks/Train_Extractor_Colab.ipynb

## Train

```bash
PYTHONPATH=. python training/sft_adversary.py \
  --model_name unsloth/Qwen2.5-1.5B-Instruct \
  --output_dir checkpoints/sft_adversary
```

## Eval (with trained extractor + trained adversary)

```bash
PYTHONPATH=. python evaluation/run_eval.py \
  --model_path checkpoints/sft_warmup \
  --adversary_model_path checkpoints/sft_adversary \
  --num_episodes 50
```

## Load (Transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "HardikJha/adversary-aea"
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, device_map="auto")
```

## Evidence

Upload `trainer_log_history.json` and `plots/sft_adversary_loss.png` from local/Colab training to this repo for judges.

## Citation

Same stack as the extractor model: Unsloth + TRL SFTTrainer; cite TRL / Unsloth / Qwen as appropriate for your writeup.
