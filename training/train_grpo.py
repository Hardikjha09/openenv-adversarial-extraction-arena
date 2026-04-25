"""
Main entry point for Training Harness.
Executes SFT Warmup followed by GRPO.
"""

import os
from training.sft_warmup import run_sft_warmup
from training.grpo_trainer import run_grpo_training

def main():
    print("=== Adversarial Structured-Extraction Arena Training Harness ===")
    
    os.makedirs("checkpoints/sft_warmup", exist_ok=True)
    os.makedirs("checkpoints/grpo_extractor", exist_ok=True)
    
    print("\n[1/2] Starting SFT Warmup Phase...")
    # Uncomment to actually run in Colab/GPU environment
    # run_sft_warmup(
    #     model_name="unsloth/Qwen2.5-1.5B-Instruct",
    #     output_dir="checkpoints/sft_warmup"
    # )
    print("SFT Warmup configured. (Execution mocked for local CPU run)")
    
    print("\n[2/2] Starting GRPO Phase...")
    # Uncomment to actually run in Colab/GPU environment
    # run_grpo_training(
    #     model_name="checkpoints/sft_warmup",
    #     output_dir="checkpoints/grpo_extractor"
    # )
    print("GRPO Trainer configured. (Execution mocked for local CPU run)")
    
    print("\n=== Training Harness Setup Complete ===")

if __name__ == "__main__":
    main()
