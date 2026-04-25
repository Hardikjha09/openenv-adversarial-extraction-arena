"""
SFT warmup for the Adversary policy: learn to emit valid edit JSON under budget.

Labels are synthesized from the corpus (heuristic targets). Same Unsloth+TRL stack as
`training/sft_warmup.py` for the Extractor.
"""

import argparse
import json
import os
import random
import re
from typing import Any, Dict, List

import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

from training.prompts import ADVERSARY_SYSTEM_PROMPT

PERTURBATION_BUDGET_DEFAULT = 200


def _rename_candidates(text: str, required: List[str]) -> List[str]:
    out: List[str] = []
    for name in required:
        if not name:
            continue
        if re.search(r"\b" + re.escape(str(name)) + r"\b", text, re.IGNORECASE):
            out.append(str(name))
    return out


def synthesize_adversary_edits(doc: Dict[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    """Build a valid edit program (dicts only) that the executor can apply."""
    text = doc.get("text") or ""
    schema = doc.get("schema") or {}
    required = list(schema.get("required") or [])
    props = list((schema.get("properties") or {}).keys())

    rename_ok = _rename_candidates(text, required)
    r = rng.random()

    edits: List[Dict[str, Any]] = []

    if r < 0.38 and rename_ok:
        old = rng.choice(rename_ok)
        new_name = old + "_Ref" if not old.endswith("_Ref") else old + "2"
        edits.append({"edit_type": "rename_field", "params": {"old_name": old, "new_name": new_name}})
    elif r < 0.58:
        edits.append(
            {"edit_type": "ocr_noise", "params": {"intensity": round(rng.uniform(0.12, 0.38), 2)}}
        )
    elif r < 0.78 and props:
        fld = rng.choice(props)
        edits.append({"edit_type": "swap_type", "params": {"field": fld, "new_type": "string"}})
    else:
        edits.append(
            {
                "edit_type": "inject_distractor",
                "params": {"content": "INTERNAL VOID — NOT AUTHORITATIVE FOR EXTRACTION"},
            }
        )

    if rng.random() < 0.28 and edits[-1]["edit_type"] != "ocr_noise":
        edits.append({"edit_type": "ocr_noise", "params": {"intensity": round(rng.uniform(0.08, 0.2), 2)}})

    return edits


def create_adversary_sft_dataset(
    corpus_path: str = "data/corpus.json",
    seed: int = 3407,
    start_idx: int = 200,
    n_docs: int = 200,
) -> Dataset:
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    rng = random.Random(seed)
    end = min(len(corpus), start_idx + n_docs)
    slice_docs = corpus[start_idx:end]
    if len(slice_docs) < n_docs and start_idx > 0:
        slice_docs = corpus[:n_docs]

    rows = []
    for doc in slice_docs:
        budget = PERTURBATION_BUDGET_DEFAULT
        prompt = ADVERSARY_SYSTEM_PROMPT.format(
            budget=budget,
            document=doc["text"],
            schema=json.dumps(doc["schema"], indent=2),
        )
        target_list = synthesize_adversary_edits(doc, rng)
        response = f"```json\n{json.dumps(target_list, indent=2)}\n```"
        text = f"{prompt}\n\n{response}"
        rows.append({"text": text})

    return Dataset.from_list(rows)


def run_sft_adversary(
    model_name: str = "unsloth/Qwen2.5-1.5B-Instruct",
    output_dir: str = "checkpoints/sft_adversary",
    corpus_path: str = "data/corpus.json",
    start_idx: int = 200,
    n_docs: int = 200,
):
    dataset = create_adversary_sft_dataset(
        corpus_path=corpus_path, seed=3407, start_idx=start_idx, n_docs=n_docs
    )

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
        dataset_text_field="text",
        formatting_func=lambda x: x["text"],
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
    try:
        with open(os.path.join(output_dir, "trainer_log_history.json"), "w", encoding="utf-8") as f:
            json.dump(trainer.state.log_history, f, indent=2)
    except Exception:
        pass
    print(f"Adversary SFT complete. Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="unsloth/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_dir", default="checkpoints/sft_adversary")
    parser.add_argument("--corpus_path", default="data/corpus.json")
    parser.add_argument("--start_idx", type=int, default=200, help="Corpus slice start (default 200 to not overlap extractor SFT [:200])")
    parser.add_argument("--n_docs", type=int, default=200)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_sft_adversary(
        model_name=args.model_name,
        output_dir=args.output_dir,
        corpus_path=args.corpus_path,
        start_idx=args.start_idx,
        n_docs=args.n_docs,
    )
