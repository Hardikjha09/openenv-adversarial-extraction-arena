"""
Evaluation Harness to evaluate checkpoints on holdout sets and generate metrics for plotting.
"""

import os
import json
import random
import argparse
import re
from typing import Any, Dict, Optional

from evaluation.elo import EloRater
from env.extraction_env import AdversarialExtractionEnv, EnvAction
from env.models import AdversaryAction, ExtractorAction, AdversaryEdit
from training.prompts import EXTRACTOR_SYSTEM_PROMPT


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    # Prefer fenced json blocks (matches training format)
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # Fallback: first JSON object-looking substring
    m2 = re.search(r"(\{[\s\S]*\})", text)
    if m2:
        try:
            return json.loads(m2.group(1))
        except Exception:
            pass

    return {}


def _load_extractor(model_path: str):
    """
    Load a local checkpoint for inference.
    Works with either:
    - `checkpoints/sft_warmup` (LoRA adapter saved by Unsloth), or
    - a full HF model path.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else None,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return model, tokenizer, device


def _run_extractor(model, tokenizer, device: str, document_text: str, schema: Dict[str, Any], max_new_tokens: int) -> Dict[str, Any]:
    import torch

    prompt = EXTRACTOR_SYSTEM_PROMPT.format(document=document_text, schema=json.dumps(schema, indent=2))
    messages = [{"role": "user", "content": prompt}]

    if hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        input_text = prompt

    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return _extract_json_from_text(decoded)


def simulate_eval_run(
    num_episodes: int = 100,
    model_path: Optional[str] = None,
    max_new_tokens: int = 512,
    save_every: int = 5,
):
    print(f"Running Evaluation Harness over {num_episodes} episodes...")
    env = AdversarialExtractionEnv(split="holdout")
    elo = EloRater()
    
    results = []

    model = tokenizer = device = None
    if model_path:
        print(f"Loading extractor checkpoint from: {model_path}")
        model, tokenizer, device = _load_extractor(model_path)
    
    for i in range(num_episodes):
        t0 = time.time()
        obs = env.reset()
        # Mock Adversary Action (Random policy baseline)
        if random.random() < 0.5:
            # Add simple ocr noise edit
            edit = AdversaryEdit(edit_type="ocr_noise", params={"intensity": 0.2}, token_cost=10)
            adv_action = EnvAction(action=AdversaryAction(edits=[edit], total_token_cost=10))
        else:
            adv_action = EnvAction(action=AdversaryAction(edits=[], total_token_cost=0))
            
        env.step(adv_action)
        
        # Real Extractor Action (if model provided), else fallback to empty dict
        extracted_json: Dict[str, Any] = {}
        if model_path:
            extracted_json = _run_extractor(
                model=model,
                tokenizer=tokenizer,
                device=device,
                document_text=env.state.document_current,
                schema=env.state.schema,
                max_new_tokens=max_new_tokens,
            )
        ext_action = EnvAction(action=ExtractorAction(extracted_json=extracted_json))
        final_obs = env.step(ext_action)
        
        ext_reward = final_obs.observation.reward
        adv_reward = env.state.adversary_reward
        
        ext_elo, adv_elo = elo.update(ext_reward, adv_reward)
        
        results.append({
            "episode": i,
            "extractor_reward": ext_reward,
            "adversary_reward": adv_reward,
            "extractor_elo": ext_elo,
            "adversary_elo": adv_elo,
            "edits_applied": len(env.state.applied_edits)
        })

        # Incremental save + progress so runs don't look stuck
        if (i + 1) % save_every == 0 or (i + 1) == num_episodes:
            os.makedirs("data", exist_ok=True)
            with open("data/eval_metrics.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            dt = time.time() - t0
            print(
                f"[{i+1}/{num_episodes}] ext_reward={ext_reward:.3f} adv_reward={adv_reward:.3f} "
                f"ext_elo={ext_elo:.1f} adv_elo={adv_elo:.1f} step_time={dt:.1f}s",
                flush=True,
            )
        
    print(f"Evaluation complete. Extractor Elo: {ext_elo:.1f}, Adversary Elo: {adv_elo:.1f}")
    print("Metrics saved to data/eval_metrics.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained extractor checkpoint (e.g. checkpoints/sft_warmup)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()
    simulate_eval_run(
        num_episodes=args.num_episodes,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        save_every=args.save_every,
    )
