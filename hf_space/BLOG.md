# Adversarial Structured-Extraction Arena: train an extractor while an adversary attacks your OCR and schema

*OpenEnv India Hackathon 2026 · Theme: multi-agent / robust structured extraction*

---

## Why this problem is hard

Real documents are **messy**: OCR confuses characters (`0`/`O`, `1`/`l`), field names **drift**, and noise can break pipelines that assume clean text and a fixed schema. Static benchmarks rarely model an **active opponent** that perturbs inputs under **rules and a budget**.

We built a **trainable two-agent arena** on **OpenEnv**: one policy **extracts** JSON under a target schema; another **adversary** proposes **executable edits** to the document and schema so the extractor must stay robust.

---

## What the environment does

**Adversarial Structured-Extraction Arena** is declared in [`openenv.yaml`](https://github.com/Hardikjha09/openenv-adversarial-extraction-arena/blob/main/openenv.yaml) and implemented in code as:

- **`AdversarialExtractionEnv`** (`env/extraction_env.py`) — stepping, observations, and rubric-based rewards.
- **Adversary executor** (`env/adversary.py`) — applies structured edits such as `rename_field`, `ocr_noise`, `swap_type`, `inject_distractor`, and more, within a **token budget**.
- **Rubric** (`env/rubric.py`) + **grader** (`grader/`) — scoring is **not** “exact string match only”; gates include valid JSON and schema coverage, with **fuzzy / typed** alignment to gold answers and bonuses for drift awareness.

The stack uses **`openenv-core`** and standard Python tooling; training uses **TRL** and **Unsloth** on **Qwen2.5-1.5B-Instruct** with **LoRA** adapters published on the Hub.

---

## Agents: Extractor (E) and Adversary (A)

| Agent | Role | Output |
|--------|------|--------|
| **E (Extractor)** | Read the (possibly perturbed) document and schema | `ExtractorAction` with **`extracted_json`** as a JSON object (training uses markdown-fenced JSON) |
| **A (Adversary)** | Spend budget to stress the extractor | `AdversaryAction` with a **list of edits** the environment applies in order |

Eval and the Space demo run **paired inference**: the adversary proposes edits, the environment applies them, then the extractor sees the stressed document/schema.

---

## How we trained (real runs, reproducible)

1. **Corpus** — Synthetic **Indian-context** documents and schemas from `data/generator.py` → `data/corpus.json` (generated in Colab or locally; not committed to git due to size).

2. **Extractor SFT** — `training/sft_warmup.py` (TRL `SFTTrainer` + Unsloth). Saves a LoRA adapter (e.g. `checkpoints/sft_warmup`) and **`trainer_log_history.json`**.

3. **Adversary SFT** — `training/sft_adversary.py` teaches the model to emit **valid edit JSON** matching the executor. Supervision uses **heuristic** edit programs sampled per document (same edit types as production). Default slice avoids overlapping the extractor’s first training slice (`--start_idx 200`, `--n_docs 200`).

4. **Optional RL** — `training/grpo_trainer.py` can refine the **extractor** further with GRPO; the main submission path is **SFT + eval**.

Training is packaged for judges in a **single Colab notebook** that clones the repo, installs dependencies, generates the corpus, runs both SFT jobs, and refreshes loss plots.

---

## Evaluation: proof beyond the demo

`evaluation/run_eval.py` runs many **holdout episodes** with optional **`--adversary_model_path`**, tracks extractor/adversary rewards, maintains **Elo**-style ratings, and writes **`eval_metrics.json`**. Plotting scripts under `plots/` turn logs into **loss** and **eval** figures.

This is the right place to quote **aggregate** behavior; the Gradio Space is for **interactive** intuition (and can be run with **GPU** so Hub LoRAs load in 4-bit).

---

## Evidence (training + evaluation artifacts)

All links are on the **Hugging Face model repos** so judges can verify without retraining:

**Extractor — [HardikJha/extractor-aea](https://huggingface.co/HardikJha/extractor-aea)**

- [Training loss](https://huggingface.co/HardikJha/extractor-aea/blob/main/plots/sft_loss.png)
- [Eval reward (moving average)](https://huggingface.co/HardikJha/extractor-aea/blob/main/plots/rewards.png)
- [Eval Elo](https://huggingface.co/HardikJha/extractor-aea/blob/main/plots/elo_ratings.png)
- [Eval metrics JSON](https://huggingface.co/HardikJha/extractor-aea/blob/main/eval_metrics.json)
- [SFT trainer log (raw)](https://huggingface.co/HardikJha/extractor-aea/blob/main/trainer_log_history.json)

**Adversary — [HardikJha/adversary-aea](https://huggingface.co/HardikJha/adversary-aea)**

- [Adversary SFT loss](https://huggingface.co/HardikJha/adversary-aea/blob/main/plots/sft_adversary_loss.png)
- [SFT trainer log](https://huggingface.co/HardikJha/adversary-aea/blob/main/trainer_log_history.json)

---

## Try it and reproduce

| Resource | URL |
|----------|-----|
| **Runnable Space (Gradio)** | [https://huggingface.co/spaces/HardikJha/extraction-arena](https://huggingface.co/spaces/HardikJha/extraction-arena) |
| **Training Colab** | [Open in Colab](https://colab.research.google.com/github/Hardikjha09/openenv-adversarial-extraction-arena/blob/main/notebooks/Train_Extractor_Colab.ipynb) |
| **Source code** | [GitHub: openenv-adversarial-extraction-arena](https://github.com/Hardikjha09/openenv-adversarial-extraction-arena) |

**Space tip:** enable **GPU** in Space settings for live **extractor-aea** / **adversary-aea** inference; on CPU the UI still demonstrates perturbations with manual / fallback paths.

---

## Honest limitations (what we are not claiming)

- The adversary’s SFT targets are **synthetic heuristics**, not human red-teaming or full multi-agent RL equilibrium.
- Under **extreme** OCR-style noise, **numeric** and **long ID** fields can still fail or drift; the rubric and qualitative demo both matter.
- **`TokenBudgetPenalty`** in the rubric is a shaping term; see `env/rubric.py` for exact weights and gates.

---

## Summary

We contribute an **OpenEnv-grounded**, **two-policy** extraction arena with **public LoRAs**, **logged training**, **eval curves**, and a **discoverable Hugging Face Space**, so judges can **re-run training in Colab** and **inspect real plots** on the Hub. The project is built **on** OpenEnv and TRL/Unsloth—not a one-off custom RL stack—so the community can extend adversaries, rubrics, and policies in one place.
