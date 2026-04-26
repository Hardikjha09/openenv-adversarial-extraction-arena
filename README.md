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

### Two trained policies (E + A)

- **Extractor (E)**: `training/sft_warmup.py` → `checkpoints/sft_warmup` · Hub: [HardikJha/extractor-aea](https://huggingface.co/HardikJha/extractor-aea)
- **Adversary (A)**: `training/sft_adversary.py` → `checkpoints/sft_adversary` · Hub: [HardikJha/adversary-aea](https://huggingface.co/HardikJha/adversary-aea) · model card source: `artifacts/hf_adversary_model_README.md`

Eval loads **both** when you pass `--adversary_model_path`; otherwise the adversary falls back to a small random edit baseline.

## Hugging Face artifacts (judges: start here)

### Runnable Space (discoverable)

- **Gradio Space**: https://huggingface.co/spaces/HardikJha/extraction-arena  
  Enable **GPU** in the Space settings for live inference with the Hub LoRAs (`extractor-aea` / `adversary-aea`). On CPU, the UI falls back to manual OCR noise and a placeholder extractor.

### Trained models + evidence (hosted on Hub)

- **Extractor LoRA**: https://huggingface.co/HardikJha/extractor-aea
- **Adversary LoRA**: https://huggingface.co/HardikJha/adversary-aea

**Evidence — extractor repo** ([HardikJha/extractor-aea](https://huggingface.co/HardikJha/extractor-aea)):

- **Training loss plot**: https://huggingface.co/HardikJha/extractor-aea/blob/main/plots/sft_loss.png
- **Eval reward plot**: https://huggingface.co/HardikJha/extractor-aea/blob/main/plots/rewards.png
- **Eval Elo plot**: https://huggingface.co/HardikJha/extractor-aea/blob/main/plots/elo_ratings.png
- **Eval metrics JSON**: https://huggingface.co/HardikJha/extractor-aea/blob/main/eval_metrics.json
- **SFT trainer log (raw)**: https://huggingface.co/HardikJha/extractor-aea/blob/main/trainer_log_history.json

**Evidence — adversary repo** ([HardikJha/adversary-aea](https://huggingface.co/HardikJha/adversary-aea)):

- **Adversary SFT loss plot**: https://huggingface.co/HardikJha/adversary-aea/blob/main/plots/sft_adversary_loss.png
- **Adversary SFT trainer log** (also inside uploaded adapter folder): https://huggingface.co/HardikJha/adversary-aea/blob/main/trainer_log_history.json

## Re-run training (Colab notebook)

Open in Colab (no “Open in Colab” button required):

- **Colab notebook**: https://colab.research.google.com/github/Hardikjha09/openenv-adversarial-extraction-arena/blob/main/notebooks/Train_Extractor_Colab.ipynb

What it does:

- clones this repo
- installs `requirements.txt`
- generates `data/corpus.json` (not committed to git; large)
- runs **extractor SFT** via `training/sft_warmup.py` and **adversary SFT** via `training/sft_adversary.py`
- generates `plots/sft_loss.png` and `plots/sft_adversary_loss.png` via `plots/generate_training_plots.py`

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
# Extractor SFT (writes checkpoints/sft_warmup + trainer_log_history.json)
PYTHONPATH=. python training/sft_warmup.py --model_name unsloth/Qwen2.5-1.5B-Instruct --output_dir checkpoints/sft_warmup

# Adversary SFT (writes checkpoints/sft_adversary + trainer_log_history.json)
PYTHONPATH=. python training/sft_adversary.py --model_name unsloth/Qwen2.5-1.5B-Instruct --output_dir checkpoints/sft_adversary

# Optional next step (RL): GRPO trainer (extractor)
PYTHONPATH=. python training/grpo_trainer.py --model_name checkpoints/sft_warmup --output_dir checkpoints/grpo_extractor
```

4) Evaluate with trained checkpoints (real inference loop):

```bash
PYTHONPATH=. python evaluation/run_eval.py --model_path checkpoints/sft_warmup --adversary_model_path checkpoints/sft_adversary --num_episodes 100
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
- **`checkpoints/` is gitignored** (by design). Public weights live on **Hugging Face**. To also mirror a small LoRA in this repo, use `git add -f checkpoints/sft_adversary/` (watch GitHub’s file-size limits; use HF for large files).
- **Secrets**: never commit tokens. This repo `.gitignore` includes `.env`.

## Additional writeups

- **Hackathon blog (markdown, in repo)**: [blog/hf_blog.md](https://github.com/Hardikjha09/openenv-adversarial-extraction-arena/blob/main/blog/hf_blog.md)
- **Same blog on the Space (for form / judges)** — after you sync `hf_space/`: [BLOG.md on the Space](https://huggingface.co/spaces/HardikJha/extraction-arena/blob/main/BLOG.md) *(URL works once `hf_space/BLOG.md` is uploaded to the Hub)*

