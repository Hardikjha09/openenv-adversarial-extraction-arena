# We trained an agent to stress-test structured document extraction — and a second agent to survive it.

*A Meta OpenEnv Hackathon Round 2 Submission*

## The problem

Large language models fail in structured extraction when documents contain schema drift, OCR noise, or injected distractors. Static benchmarks miss that dynamic failure mode. We need an environment where an **adversary** perturbs inputs under rules and a budget, and an **extractor** is scored on **valid, schema-aligned JSON**.

## What we built: Adversarial Structured-Extraction Arena

An **OpenEnv-style** environment with two agents:

- **Agent E (Extractor):** Reads messy documents and outputs JSON matching a target schema (markdown-fenced JSON, rubric-scored).
- **Agent A (Adversary):** Proposes a list of executable edits (`rename_field`, `ocr_noise`, `swap_type`, …) within a **token budget**, applied by the environment.

## How we trained

- **Supervised fine-tuning (SFT)** on a synthetic Indian-context corpus: both agents use **TRL + Unsloth** with **Qwen2.5-1.5B-Instruct** and **LoRA** adapters (`extractor-aea`, `adversary-aea` on Hugging Face).
- **Extractor:** SFT on extraction prompts; the repo also includes an **optional GRPO** stage (`training/grpo_trainer.py`) for further policy improvement.
- **Adversary:** SFT with **heuristic** edit targets sampled from the same corpus (valid edit programs the executor can apply)—a strong **warm start** for arena evaluation, not a human-in-the-loop attack model.

## Results

The evaluation harness (`evaluation/run_eval.py`) runs paired extractor + adversary inference over holdout episodes, tracks rewards, and exports metrics/plots. After training, the extractor maintains **robust JSON validity and rubric scores** under learned adversary policies compared to random baselines.

**Try it:** [Hugging Face Space — Extraction Arena](https://huggingface.co/spaces/HardikJha/extraction-arena) (enable **GPU** in Space settings for live LoRA inference).

**Reproduce training:** [Colab notebook](https://colab.research.google.com/github/Hardikjha09/openenv-adversarial-extraction-arena/blob/main/notebooks/Train_Extractor_Colab.ipynb) · **Code:** [GitHub](https://github.com/Hardikjha09/openenv-adversarial-extraction-arena)
