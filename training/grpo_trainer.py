"""
GRPO Training Script using OpenEnv, TRL, and Unsloth.
"""

import os
import json
import torch
import random
from typing import List, Dict, Any
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel, PatchFastRL
from openenv.core.env_server import create_fastapi_app
from env.extraction_env import AdversarialExtractionEnv, EnvAction
from training.prompts import EXTRACTOR_SYSTEM_PROMPT

# Patch Unsloth for RL
PatchFastRL("GRPO", FastLanguageModel)

def build_grpo_dataset(corpus_path: str = "data/corpus.json", num_samples: int = 1000) -> Dataset:
    """Build dataset for GRPO. Prompts are Extractor inputs."""
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
        
    data = []
    # Skip the first 200 used for SFT
    for doc in corpus[200:200+num_samples]:
        prompt = EXTRACTOR_SYSTEM_PROMPT.format(
            document=doc["text"],
            schema=json.dumps(doc["schema"], indent=2)
        )
        
        # Extractor only needs the prompt, no response in RL dataset
        messages = [{"role": "user", "content": prompt}]
        data.append({"prompt": messages, "env_data": {"doc": doc}})
        
    return Dataset.from_list(data)

def reward_extractor(prompts, completions, **kwargs) -> List[float]:
    """
    Reward function that uses OpenEnv environment logic.
    For standard offline GRPO we can instantiate the rubric directly,
    or use the running env if it's an online RL setup.
    We will use the rubric offline here to score completions.
    """
    from env.rubric import build_extractor_rubric
    from env.models import EpisodeState, ExtractorAction
    
    rubric = build_extractor_rubric(token_budget_mode="linear")
    env_data = kwargs.get("env_data", [])
    
    rewards = []
    for i, completion in enumerate(completions):
        try:
            doc_data = env_data[i]["doc"]
            # Extract JSON from completion text
            comp_text = completion[0]["content"] if isinstance(completion, list) else completion
            
            # Very basic extraction logic
            import re
            json_match = re.search(r"```json\n(.*?)\n```", comp_text, re.DOTALL)
            extracted_json = {}
            if json_match:
                extracted_json = json.loads(json_match.group(1))
            
            state = EpisodeState(
                episode_id="eval",
                document_original=doc_data["text"],
                document_current=doc_data["text"],
                schema=doc_data["schema"],
                gold_answers=doc_data["gold"],
                doc_type=doc_data["type"]
            )
            state.extractor_action = ExtractorAction(extracted_json=extracted_json)
            state.metadata = {"completion_tokens": len(comp_text.split())} # Rough approximation
            
            reward = rubric(state)
            rewards.append(reward)
            
        except Exception as e:
            rewards.append(0.0)
            
    return rewards

def run_grpo_training(
    model_name: str = "checkpoints/sft_warmup", 
    output_dir: str = "checkpoints/grpo_extractor"
):
    dataset = build_grpo_dataset()
    
    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=100,
        logging_steps=10,
        optim="adamw_8bit",
        seed=3407,
        report_to="none", # Set to wandb for actual tracking
    )
    
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_extractor],
        args=training_args,
        train_dataset=dataset,
    )
    
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"GRPO Training complete. Model saved to {output_dir}")

if __name__ == "__main__":
    os.makedirs("checkpoints/grpo_extractor", exist_ok=True)
    print("GRPO Trainer script ready. Execute `run_grpo_training()` to begin.")
