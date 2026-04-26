"""
Lazy-load PEFT adapters for the Space demo: only one role (extractor or adversary) on GPU at a time.
Requires a GPU Space for real inference; CPU runtimes fall back to manual / placeholder paths in app.py.
"""

from __future__ import annotations

import gc
import json
import os
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from env.models import AdversaryEdit, EditType
from prompts import ADVERSARY_SYSTEM_PROMPT, EXTRACTOR_SYSTEM_PROMPT

Role = Literal["extractor", "adversary"]

PERTURBATION_BUDGET_DEFAULT = 200

DEFAULT_BASE = os.environ.get("SPACE_BASE_MODEL", "unsloth/Qwen2.5-1.5B-Instruct")
EXTRACTOR_ADAPTER = os.environ.get("SPACE_EXTRACTOR_ADAPTER", "HardikJha/extractor-aea")
ADVERSARY_ADAPTER = os.environ.get("SPACE_ADVERSARY_ADAPTER", "HardikJha/adversary-aea")

_state: Dict[str, Any] = {
    "role": None,
    "model": None,
    "tokenizer": None,
    "last_error": None,
}


def backend_status_message() -> str:
    err = _state.get("last_error")
    if err:
        return f"Model backend: error — {err}"
    if not torch.cuda.is_available():
        return (
            "Model backend: no CUDA — enable a GPU in Space Settings for live "
            "`extractor-aea` / `adversary-aea` inference (CPU fallback uses manual OCR + placeholder extract)."
        )
    if _state["model"] is None:
        return "Model backend: ready (GPU). First inference will load the requested LoRA (~10–40s)."
    return f"Model backend: loaded `{_state['role']}` on GPU."


def _clear_model() -> None:
    _state["model"] = None
    _state["tokenizer"] = None
    _state["role"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_peft(role: Role) -> Tuple[Any, Any]:
    """Load base + one adapter in 4-bit on CUDA."""
    adapter_id = EXTRACTOR_ADAPTER if role == "extractor" else ADVERSARY_ADAPTER
    tok = AutoTokenizer.from_pretrained(adapter_id, trust_remote_code=True)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    use_token = hf_token if hf_token else None

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        DEFAULT_BASE,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        token=use_token,
    )
    model = PeftModel.from_pretrained(base, adapter_id, token=use_token)
    model.eval()
    return model, tok


def ensure_role(role: Role) -> Tuple[Optional[Any], Optional[Any]]:
    """Return (model, tokenizer) for `role`, swapping out the other adapter if needed."""
    _state["last_error"] = None
    if not torch.cuda.is_available():
        return None, None
    if _state["role"] == role and _state["model"] is not None:
        return _state["model"], _state["tokenizer"]
    try:
        _clear_model()
        model, tokenizer = _load_peft(role)
        _state["model"] = model
        _state["tokenizer"] = tokenizer
        _state["role"] = role
        return model, tokenizer
    except Exception as e:
        _state["last_error"] = str(e)
        _clear_model()
        return None, None


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m2 = re.search(r"(\{[\s\S]*\})", text)
    if m2:
        try:
            return json.loads(m2.group(1))
        except Exception:
            pass
    return {}


def _extract_json_array_from_text(text: str) -> List[Any]:
    m = re.search(r"```json\s*(\[[\s\S]*?\])\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            out = json.loads(m.group(1))
            return out if isinstance(out, list) else []
        except Exception:
            pass
    m2 = re.search(r"(\[[\s\S]*\])", text)
    if m2:
        try:
            out = json.loads(m2.group(1))
            return out if isinstance(out, list) else []
        except Exception:
            pass
    return []


def _clip_edits_to_budget(edits: List[AdversaryEdit], budget: int) -> List[AdversaryEdit]:
    out: List[AdversaryEdit] = []
    used = 0
    for e in edits:
        if used + e.token_cost <= budget:
            out.append(e)
            used += e.token_cost
    return out


def _edits_from_model_text(decoded: str, budget: int) -> List[AdversaryEdit]:
    raw_list = _extract_json_array_from_text(decoded)
    edits: List[AdversaryEdit] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        et = item.get("edit_type")
        params = item.get("params") if isinstance(item.get("params"), dict) else {}
        if et is None:
            continue
        try:
            edits.append(
                AdversaryEdit(edit_type=EditType(str(et)), params=params, token_cost=0)
            )
        except Exception:
            continue
    return _clip_edits_to_budget(edits, budget)


def _model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def run_adversary(
    document_text: str,
    schema: Dict[str, Any],
    budget: int = PERTURBATION_BUDGET_DEFAULT,
    max_new_tokens: int = 384,
) -> List[AdversaryEdit]:
    model, tokenizer = ensure_role("adversary")
    if model is None or tokenizer is None:
        return []

    prompt = ADVERSARY_SYSTEM_PROMPT.format(
        budget=budget,
        document=document_text,
        schema=json.dumps(schema, indent=2),
    )
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        input_text = prompt

    inputs = tokenizer(input_text, return_tensors="pt")
    dev = _model_device(model)
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return _edits_from_model_text(decoded, budget)


def run_extractor(
    document_text: str,
    schema: Dict[str, Any],
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    model, tokenizer = ensure_role("extractor")
    if model is None or tokenizer is None:
        return {}

    prompt = EXTRACTOR_SYSTEM_PROMPT.format(
        document=document_text,
        schema=json.dumps(schema, indent=2),
    )
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        input_text = prompt

    inputs = tokenizer(input_text, return_tensors="pt")
    dev = _model_device(model)
    inputs = {k: v.to(dev) for k, v in inputs.items()}

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
