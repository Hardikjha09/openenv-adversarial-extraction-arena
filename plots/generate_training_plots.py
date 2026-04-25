"""
Generate plots from real training logs saved by TRL trainers.
"""

import json
import os
from typing import List, Dict, Any, Optional

import matplotlib.pyplot as plt


def _load_log_history(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_series(log_history: List[Dict[str, Any]], key: str):
    xs, ys = [], []
    for rec in log_history:
        if key in rec and "step" in rec:
            xs.append(rec["step"])
            ys.append(rec[key])
    return xs, ys


def plot_training_loss(
    log_path: str,
    output_path: str,
    title: str,
    loss_key_candidates: Optional[List[str]] = None,
):
    if loss_key_candidates is None:
        loss_key_candidates = ["loss", "train_loss"]

    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    log_history = _load_log_history(log_path)
    chosen = None
    for k in loss_key_candidates:
        xs, ys = _extract_series(log_history, k)
        if xs:
            chosen = (k, xs, ys)
            break

    if chosen is None:
        print(f"No loss keys found in {log_path}")
        return

    key, xs, ys = chosen
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, linewidth=2)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(key)
    plt.grid(True, linestyle="--", alpha=0.6)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    print(f"Saved {title} plot to {output_path}")


if __name__ == "__main__":
    # Defaults align to repo checkpoints
    plot_training_loss(
        log_path="checkpoints/sft_warmup/trainer_log_history.json",
        output_path="plots/sft_loss.png",
        title="SFT Warmup Training Loss",
    )
    plot_training_loss(
        log_path="checkpoints/grpo_extractor/trainer_log_history.json",
        output_path="plots/grpo_loss.png",
        title="GRPO Training (Logged Metric)",
    )
    plot_training_loss(
        log_path="checkpoints/sft_adversary/trainer_log_history.json",
        output_path="plots/sft_adversary_loss.png",
        title="Adversary SFT Warmup Training Loss",
    )

