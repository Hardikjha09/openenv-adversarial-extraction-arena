"""
Helper script for Colab execution.
Auto-detects T4 vs A100 and sets optimized Unsloth parameters.
"""

import torch
import os
from training.train_grpo import main as run_training

def setup_colab_environment():
    print("=== Colab Environment Setup ===")
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not detected. Ensure you are using a GPU runtime.")
        return
        
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Detected GPU: {gpu_name}")
    
    if "A100" in gpu_name:
        print("A100 detected. Optimizing for bfloat16 and larger batches.")
        os.environ["TORCH_DTYPE"] = "bfloat16"
        # Settings for A100 could be injected into config here
    elif "T4" in gpu_name:
        print("T4 detected. Optimizing for float16 and gradient accumulation.")
        os.environ["TORCH_DTYPE"] = "float16"
    else:
        print("Generic GPU detected. Defaulting to float16.")
        os.environ["TORCH_DTYPE"] = "float16"

if __name__ == "__main__":
    setup_colab_environment()
    print("Launching training pipeline...")
    run_training()
