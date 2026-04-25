"""
Main entry point for Training Harness.
Executes SFT Warmup followed by GRPO.
"""

import os
import argparse
from training.sft_warmup import run_sft_warmup
from training.grpo_trainer import run_grpo_training

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="unsloth/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--run_sft", action="store_true", help="Run SFT warmup")
    parser.add_argument("--run_grpo", action="store_true", help="Run GRPO after SFT")
    parser.add_argument("--sft_out", default="checkpoints/sft_warmup")
    parser.add_argument("--grpo_out", default="checkpoints/grpo_extractor")
    args = parser.parse_args()

    print("=== Adversarial Structured-Extraction Arena Training Harness ===")
    os.makedirs(args.sft_out, exist_ok=True)
    os.makedirs(args.grpo_out, exist_ok=True)

    if args.run_sft:
        print("\n[1/2] Running SFT Warmup Phase...")
        run_sft_warmup(model_name=args.base_model, output_dir=args.sft_out)
    else:
        print("\n[1/2] Skipping SFT Warmup (pass --run_sft to enable).")

    if args.run_grpo:
        print("\n[2/2] Running GRPO Phase...")
        run_grpo_training(model_name=args.sft_out, output_dir=args.grpo_out)
    else:
        print("\n[2/2] Skipping GRPO (pass --run_grpo to enable).")

    print("\n=== Training Harness Complete ===")

if __name__ == "__main__":
    main()
