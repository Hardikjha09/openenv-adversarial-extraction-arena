"""
Supervised Fine-Tuning (SFT) Warmup.
We use Unsloth to quickly fine-tune the model on synthetic perfect responses
so it understands the expected JSON structure before RL.
"""

import os
import json
import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from training.prompts import EXTRACTOR_SYSTEM_PROMPT

def create_sft_dataset(corpus_path: str = "data/corpus.json") -> Dataset:
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
        
    data = []
    # Take first 200 documents for SFT warmup
    for doc in corpus[:200]:
        prompt = EXTRACTOR_SYSTEM_PROMPT.format(
            document=doc["text"],
            schema=json.dumps(doc["schema"], indent=2)
        )
        
        response = f"```json\n{json.dumps(doc['gold'], indent=2)}\n```"
        
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        data.append({"messages": messages})
        
    return Dataset.from_list(data)

def run_sft_warmup(model_name: str = "unsloth/Qwen2.5-1.5B-Instruct", output_dir: str = "checkpoints/sft_warmup"):
    dataset = create_sft_dataset()
    
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
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="messages",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=50,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            report_to="none",
        ),
    )
    
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"SFT Warmup complete. Model saved to {output_dir}")

if __name__ == "__main__":
    # Ensure checkpoints directory exists
    os.makedirs("checkpoints/sft_warmup", exist_ok=True)
    # We don't run it by default on standard execution unless explicitly called
    print("SFT Warmup script ready. Execute `run_sft_warmup()` to begin.")
