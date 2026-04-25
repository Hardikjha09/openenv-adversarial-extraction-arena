# Adversarial Structured-Extraction Arena (OpenEnv)

**Meta OpenEnv Hackathon — Round 2** · **Theme 1: Multi-Agent Interactions**

**One-liner:** A two-agent OpenEnv environment where an **Adversary** perturbs messy Indian-context documents/schemas (under a token budget) and an **Extractor** must output **valid JSON** matching a target schema.

## Why this problem matters

Structured extraction from “real world” documents fails when text is noisy (OCR), fields drift (renames/type swaps), and distractors appear. This repo turns that failure mode into a **trainable environment**: you can measure robustness with a rubric, compare policies, and iterate.

## What we built (OpenEnv-first)

This repo is intentionally built **on OpenEnv** (see `openenv.yaml`) rather than inventing a bespoke RL loop:

- **Environment**: `env/extraction_env.py` implements `AdversarialExtractionEnv` with OpenEnv-style stepping, rubric scoring, and FastAPI hosting (`env/server.py`).
- **Agents**:
  - **Extractor (E)**: produces `ExtractorAction(extracted_json=...)`.
  - **Adversary (A)**: proposes `AdversaryAction(edits=[...])` executed by `env/adversary.py`.
- **Scoring**: `env/rubric.py` + `grader/` implement fuzzy / typed scoring (not “exact string match only”).

## Hugging Face artifacts (judges: start here)

### Runnable Space (discoverable)

- **Gradio Space**: https://huggingface.co/spaces/HardikJha/extraction-arena

### Trained model + evidence (hosted on Hub)

- **Model repo (LoRA / SFT warmup checkpoint)**: https://huggingface.co/HardikJha/extractor-aea

**Evidence files (direct links):**

- **Training loss plot**: https://huggingface.co/HardikJha/extractor-aea/blob/main/plots/sft_loss.png
- **Eval reward plot**: https://huggingface.co/HardikJha/extractor-aea/blob/main/plots/rewards.png
- **Eval Elo plot**: https://huggingface.co/HardikJha/extractor-aea/blob/main/plots/elo_ratings.png
- **Eval metrics JSON**: https://huggingface.co/HardikJha/extractor-aea/blob/main/eval_metrics.json
- **SFT trainer log (raw)**: https://huggingface.co/HardikJha/extractor-aea/blob/main/trainer_log_history.json

## Re-run training (Colab notebook)

Open in Colab (no “Open in Colab” button required):

- **Colab notebook**: https://colab.research.google.com/github/Hardikjha09/openenv-adversarial-extraction-arena/blob/main/notebooks/Train_Extractor_Colab.ipynb

What it does:

- clones this repo
- installs `requirements.txt`
- generates `data/corpus.json` (not committed to git; large)
- runs **real SFT warmup** via `training/sft_warmup.py`
- generates `plots/sft_loss.png` via `plots/generate_training_plots.py`

## Local reproduction (GPU recommended)

**Prereqs:** Python 3.10+ and a CUDA GPU (training is not practical on CPU).

1) Install deps:

```bash
pip install -r requirements.txt
```

2) Generate corpus (required because `data/corpus.json` is gitignored):

```bash
PYTHONPATH=. python data/generator.py
```

3) Train (real):

```bash
# SFT warmup (writes checkpoints/sft_warmup + trainer_log_history.json)
PYTHONPATH=. python training/sft_warmup.py --model_name unsloth/Qwen2.5-1.5B-Instruct --output_dir checkpoints/sft_warmup

# Optional next step (RL): GRPO trainer
PYTHONPATH=. python training/grpo_trainer.py --model_name checkpoints/sft_warmup --output_dir checkpoints/grpo_extractor
```

4) Evaluate with the trained checkpoint (real inference loop):

```bash
PYTHONPATH=. python evaluation/run_eval.py --model_path checkpoints/sft_warmup --num_episodes 100
PYTHONPATH=. python plots/generate_plots.py
PYTHONPATH=. python plots/generate_training_plots.py
```

5) Run the OpenEnv FastAPI server:

```bash
PYTHONPATH=. python env/server.py
```

6) Run the local Gradio demo:

```bash
PYTHONPATH=. python demo/app.py
```

## Docker (optional)

```bash
docker-compose up --build
```

## Repo map

```text
adversarial-extraction-arena/
├── data/                  # synthetic generator + corpus loader
├── env/                   # OpenEnv env + models + rubric + server
├── grader/                # fuzzy / typed scoring helpers
├── training/              # TRL + Unsloth training scripts
├── evaluation/            # eval harness + Elo
├── demo/                  # Gradio demo
├── notebooks/             # Colab entrypoint
├── plots/                 # plotting scripts
├── hf_space/              # Space bundle (optional)
├── blog/                  # narrative writeup (markdown)
└── openenv.yaml           # OpenEnv declaration
```

## Notes / gotchas

- **`data/corpus.json` is not in git** (by design). You must generate it before training/eval.
- **`checkpoints/` is not in git** (by design). Train in Colab or locally, then upload to HF if you want a public checkpoint.
- **Secrets**: never commit tokens. This repo `.gitignore` includes `.env`.

## Additional writeups

- **Project blog (markdown)**: `blog/hf_blog.md`

