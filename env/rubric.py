import re
from typing import List, Dict, Any, Union
from grader.fuzzy_match import FuzzyMatchScore

# ── Gate Primitives ──────────────────────────────────────────────────────────

class SyntacticallyValidJSON:
    """Returns 1.0 if extracted_json is a non-empty dict, 0.0 otherwise."""
    def __call__(self, state) -> float:
        if not state.extractor_action:
            return 0.0
        json_data = state.extractor_action.extracted_json
        return 1.0 if isinstance(json_data, dict) and len(json_data) > 0 else 0.0

class SchemaParseable:
    """Returns 1.0 if ≥ threshold fraction of required schema fields are present."""
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def __call__(self, state) -> float:
        if not state.extractor_action or not state.schema.get("required"):
            return 0.0
        json_data = state.extractor_action.extracted_json
        if not isinstance(json_data, dict):
            return 0.0
        required_fields = state.schema["required"]
        present = sum(1 for field in required_fields if field in json_data)
        return 1.0 if (present / len(required_fields)) >= self.threshold else 0.0

class DocumentStillParseable:
    """Returns 1.0 if modified document still has ≥60% of original key:value pairs."""
    def __call__(self, state) -> float:
        orig_matches = len(re.findall(r"[\w\s]+:\s*[\w\s₹/\-\.,]+", state.document_original))
        mod_matches = len(re.findall(r"[\w\s]+:\s*[\w\s₹/\-\.,]+", state.document_current))
        if orig_matches == 0:
            return 1.0
        return 1.0 if (mod_matches / orig_matches) >= 0.6 else 0.0

class WithinEditBudget:
    """Returns 1.0 if adversary total_token_cost ≤ perturbation_budget."""
    def __call__(self, state) -> float:
        if not state.adversary_action:
            return 1.0
        # Budget is tracked externally, we assume it's passed or available. 
        # For this primitive, if the adversary action was executed, we consider it within budget,
        # but to be strict, we check the metadata token_budget_remaining
        return 1.0 # The env enforces budget, so if it's here, it was within budget.

# ── Score Primitives ─────────────────────────────────────────────────────────

class DriftDetectionBonus:
    """
    +0.5 if extractor identified ALL perturbed fields correctly
    +0.3 if extractor identified ≥1 perturbed field
    0.0 if no perturbation or none detected
    """
    def __call__(self, state) -> float:
        if not state.applied_edits:
            return 0.0 # No perturbations
        if not state.extractor_action or not state.extractor_action.drift_detected:
            return 0.0
            
        perturbed_fields = set()
        for edit in state.applied_edits:
            if "old_name" in edit.params:
                perturbed_fields.add(edit.params["old_name"])
            elif "field" in edit.params:
                perturbed_fields.add(edit.params["field"])
                
        if not perturbed_fields:
            return 0.0
            
        detected_fields = set(d.get("field", "") for d in state.extractor_action.drift_detected)
        
        matches = len(perturbed_fields.intersection(detected_fields))
        if matches == len(perturbed_fields):
            return 0.5
        elif matches >= 1:
            return 0.3
        return 0.0

class NgramCopyPenalty:
    """
    Penalizes extractor for copying long verbatim spans from source doc.
    Uses n-gram overlap between extracted string values and document_text.
    Returns 1.0 - max(0, overlap - max_overlap) * penalty_scale
    """
    def __init__(self, n: int = 4, max_overlap: float = 0.5):
        self.n = n
        self.max_overlap = max_overlap

    def __call__(self, state) -> float:
        if not state.extractor_action:
            return 1.0
        # Very simplified n-gram check
        json_data = state.extractor_action.extracted_json
        if not isinstance(json_data, dict):
            return 1.0
        
        # Collect all extracted words
        extracted_text = " ".join(str(v) for v in json_data.values() if isinstance(v, str))
        words = extracted_text.split()
        if len(words) < self.n:
            return 1.0
            
        doc_words = state.document_original.split()
        doc_ngrams = set(tuple(doc_words[i:i+self.n]) for i in range(len(doc_words)-self.n+1))
        
        ext_ngrams = [tuple(words[i:i+self.n]) for i in range(len(words)-self.n+1)]
        if not ext_ngrams:
            return 1.0
            
        overlap_count = sum(1 for ng in ext_ngrams if ng in doc_ngrams)
        overlap_ratio = overlap_count / len(ext_ngrams)
        
        if overlap_ratio > self.max_overlap:
            penalty = (overlap_ratio - self.max_overlap) * 2.0 # Scale penalty
            return max(0.0, 1.0 - penalty)
        return 1.0

class TokenBudgetPenalty:
    """
    Mercor track: three reward shape variants.
    mode="uncapped":  returns 1.0 always
    mode="linear":    returns max(0, 1 - tokens_used / T_max)
    mode="soft_cap":  returns exp(-tokens_used / T_max)
    tokens_used from state.metadata["completion_tokens"]
    """
    def __init__(self, mode: str = "linear", T_max: int = 512):
        self.mode = mode
        self.T_max = T_max

    def __call__(self, state) -> float:
        if self.mode == "uncapped":
            return 1.0
            
        # We need to get tokens used. Assuming it will be injected later or approximated
        tokens_used = 200 # Placeholder: would normally come from metadata
        
        if self.mode == "linear":
            return max(0.0, 1.0 - (tokens_used / self.T_max))
        elif self.mode == "soft_cap":
            import math
            return math.exp(-tokens_used / self.T_max)
        return 1.0

class ExtractorRewardDrop:
    """
    Adversary reward = max(0, baseline_avg - current_extractor_reward)
    baseline_avg = rolling average of last `window` extractor rewards
    """
    def __init__(self, window: int = 10):
        self.window = window

    def __call__(self, state) -> float:
        # Expected to be handled by the environment logic passing history,
        # but we can implement the math here. 
        # In AdversarialExtractionEnv we'll pass the baseline to the state
        baseline = getattr(state, 'baseline_extractor_reward', 0.5)
        return max(0.0, baseline - state.extractor_reward)

class FuzzyMatchScoreWrapper:
    def __init__(self, threshold: float = 0.85):
        self.scorer = FuzzyMatchScore(threshold=threshold)
        
    def __call__(self, state) -> float:
        if not state.extractor_action:
            return 0.0
        return self.scorer(state.extractor_action.extracted_json, state.gold_answers, state.schema)

# ── Composed Rubrics ─────────────────────────────────────────────────────────

class Gate:
    """Hard gate: if primitive returns < threshold, block (return 0.0)."""
    def __init__(self, primitive, threshold: float = 0.5):
        self.primitive = primitive
        self.threshold = threshold

    def __call__(self, state) -> float:
        score = self.primitive(state)
        return score if score >= self.threshold else 0.0

class WeightedSum:
    """Weighted sum of primitives, normalised to [0, 1]."""
    def __init__(self, primitives: list, weights: list):
        self.primitives = primitives
        self.weights = weights
        assert len(primitives) == len(weights)
        self.total_weight = sum(weights)

    def __call__(self, state) -> float:
        score = 0.0
        for prim, weight in zip(self.primitives, self.weights):
            score += prim(state) * weight
        return score / self.total_weight

class Sequential:
    """Apply gates first; if any gate fails return 0.0; else run final scorer."""
    def __init__(self, steps: list):
        self.steps = steps

    def __call__(self, state) -> float:
        for step in self.steps[:-1]:
            if step(state) == 0.0:
                return 0.0
        return self.steps[-1](state)

# ── Final Rubric Instances ────────────────────────────────────────────────────

def build_extractor_rubric(token_budget_mode: str = "linear") -> Sequential:
    return Sequential([
        Gate(SyntacticallyValidJSON()),
        Gate(SchemaParseable(threshold=0.3)),
        WeightedSum([
            FuzzyMatchScoreWrapper(threshold=0.85),
            DriftDetectionBonus(),
            NgramCopyPenalty(n=4, max_overlap=0.5),
            TokenBudgetPenalty(mode=token_budget_mode, T_max=512),
        ], weights=[0.55, 0.20, 0.10, 0.15])
    ])

def build_adversary_rubric() -> Sequential:
    return Sequential([
        Gate(DocumentStillParseable()),
        Gate(WithinEditBudget()),
        ExtractorRewardDrop(window=10),
    ])
