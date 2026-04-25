from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


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

    @model_validator(mode="after")
    def validate_cost(self):
        expected = EDIT_TOKEN_COSTS[self.edit_type.value]
        if self.token_cost != expected:
            self.token_cost = expected
        return self


class ExtractorAction(BaseModel):
    extracted_json: Dict[str, Any]
    drift_detected: Optional[List[Dict[str, str]]] = None  # [{"field": str, "reason": str}]
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)

