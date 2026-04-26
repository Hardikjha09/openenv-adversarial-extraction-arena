---
title: Extraction Arena
emoji: 🏢
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
---

## Running on Hugging Face Spaces

- **GPU:** Set hardware to **GPU** (Space **Settings → Hardware**). The app loads `HardikJha/extractor-aea` and `HardikJha/adversary-aea` in 4-bit (one adapter in VRAM at a time).
- **Gated models:** If the base model or adapters require auth, add a **Space secret** or variable `HF_TOKEN` with a read token.
- **Optional:** `SPACE_USE_MODELS=0` disables LoRA loading (manual / placeholder only). Override adapter IDs with `SPACE_BASE_MODEL`, `SPACE_EXTRACTOR_ADAPTER`, `SPACE_ADVERSARY_ADAPTER`.

Configuration reference: https://huggingface.co/docs/hub/spaces-config-reference
