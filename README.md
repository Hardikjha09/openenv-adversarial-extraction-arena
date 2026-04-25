# Adversarial Structured-Extraction Arena

*Meta OpenEnv Hackathon Round 2 Submission*
*Theme 1: Multi-Agent Interactions*

**Tagline:** "We trained an agent to break structured document extraction — and made the extractor unbreakable."

## Overview

This project implements a complete, production-ready, two-agent OpenEnv reinforcement learning environment. 
- **Agent E (Extractor)** reads synthetic Indian-context documents (GST Invoices, PAN Applications, FIRs, Medical Prescriptions, Land Records) and outputs structured JSON matching a predefined schema.
- **Agent A (Adversary)** perturbs those documents to break the extractor, operating within a strict token budget constraint.

Both agents are trained jointly via **GRPO** (Group Relative Policy Optimization) using TRL + Unsloth.

## Project Structure

```text
adversarial-extraction-arena/
├── data/                  # Schema definitions, synthetic generator, corpus manager
├── env/                   # OpenEnv integration, Pydantic schemas, FastAPI server, Rubric
├── grader/                # Fuzzy matching and specialized scoring
├── training/              # GRPO/TRL+Unsloth SFT and RL setup
├── evaluation/            # Holdout execution, Elo rating calculation
├── demo/                  # Gradio UI for showcasing perturbations vs. extraction
├── notebooks/             # Colab deployment scripts
├── plots/                 # Python scripts to visualize Eval results
├── blog/                  # Markdown write-up of the hackathon journey
└── hf_space/              # Files for HuggingFace Space Deployment
```

## Setup & Running

**Prerequisites:** Python 3.10+

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate synthetic corpus:
   ```bash
   PYTHONPATH=. python data/generator.py
   ```
3. Run the OpenEnv server:
   ```bash
   PYTHONPATH=. python env/server.py
   ```
4. Run training (real):
   ```bash
   # SFT warmup (recommended first run)
   PYTHONPATH=. python training/train_grpo.py --run_sft

   # Optional: GRPO after SFT
   PYTHONPATH=. python training/train_grpo.py --run_sft --run_grpo
   ```
5. Run Evaluation (real model):
   ```bash
   PYTHONPATH=. python evaluation/run_eval.py --model_path checkpoints/sft_warmup --num_episodes 100
   PYTHONPATH=. python plots/generate_plots.py
   PYTHONPATH=. python plots/generate_training_plots.py
   ```
6. Run Gradio Demo:
   ```bash
   PYTHONPATH=. python demo/app.py
   ```

## Docker (Optional)
Run the environment and demo concurrently:
```bash
docker-compose up --build
```

## Hugging Face Space

- **Space (runnable)**: `https://huggingface.co/spaces/HardikJha/extraction-arena`

## Colab (judges can re-run)

- Notebook: `notebooks/Train_Extractor_Colab.ipynb`
