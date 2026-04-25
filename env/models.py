from pydantic import BaseModel, Field, model_validator
from typing import Any, Dict, List, Literal, Optional, Tuple
from enum import Enum

class EditType(str, Enum):
    rename_field = "rename_field"
    swap_type = "swap_type"
    inject_distractor = "inject_distractor"
    mutate_format = "mutate_format"
    add_required_field = "add_required_field"
    ocr_noise = "ocr_noise"
    swap_columns = "swap_columns"

EDIT_TOKEN_COSTS = {
    "rename_field": 10,
    "swap_type": 15,
    "inject_distractor": 25,
    "mutate_format": 10,
    "add_required_field": 20,
    "ocr_noise": 5,
    "swap_columns": 15,
}

class AdversaryEdit(BaseModel):
    edit_type: EditType
    params: Dict[str, Any]
    token_cost: int

    @model_validator(mode='after')
    def validate_cost(self):
        expected = EDIT_TOKEN_COSTS[self.edit_type.value]
        if self.token_cost != expected:
            self.token_cost = expected
        return self

class ExtractorAction(BaseModel):
    extracted_json: Dict[str, Any]
    drift_detected: Optional[List[Dict[str, str]]] = None  # [{"field": str, "reason": str}]
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)

class AdversaryAction(BaseModel):
    edits: List[AdversaryEdit]
    total_token_cost: int

    @model_validator(mode='after')
    def validate_budget(self):
        self.total_token_cost = sum(e.token_cost for e in self.edits)
        return self

class ExtractionObservation(BaseModel):
    document_text: str
    target_schema: Dict[str, Any]
    perturbation_history: List[str] = []
    token_budget_remaining: int
    step: int
    reward: Optional[float] = None
    # Mercor Archipelago schema — log always
    metadata: Dict[str, Any] = Field(default_factory=lambda: {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "episode_id": "",
        "doc_type": "",
    })

class AdversaryObservation(BaseModel):
    document_text: str
    target_schema: Dict[str, Any]
    extractor_reward_history: List[float] = []
    token_budget_remaining: int
    perturbation_budget: int
    step: int
    reward: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EpisodeState(BaseModel):
    episode_id: str
    document_original: str
    document_current: str
    schema: Dict[str, Any]
    gold_answers: Dict[str, Any]
    doc_type: str
    extractor_action: Optional[ExtractorAction] = None
    adversary_action: Optional[AdversaryAction] = None
    applied_edits: List[AdversaryEdit] = []
    baseline_extractor_reward: float = 0.5
    extractor_reward: float = 0.0
    adversary_reward: float = 0.0
    done: bool = False
    step: int = 0
