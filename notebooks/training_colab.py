"""
Training Colab Notebook
(Save as .ipynb when running in Google Colab)
"""

# Install dependencies
# !pip install openenv-core "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" trl peft transformers pydantic faker rapidfuzz python-dateutil matplotlib

# Clone Repository
# !git clone https://github.com/yourusername/adversarial-extraction-arena.git
# %cd adversarial-extraction-arena

import os
import torch
import warnings
warnings.filterwarnings('ignore')

from training.train_grpo import main as run_training

if __name__ == "__main__":
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        print(f"Colab GPU detected: {gpu}")
        run_training()
    else:
        print("Please enable GPU in Runtime > Change runtime type")
