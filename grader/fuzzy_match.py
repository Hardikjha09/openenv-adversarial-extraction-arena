import re
from rapidfuzz import fuzz
from dateutil.parser import parse as parse_date
from typing import Dict, Any, Union, List

class FuzzyMatchScore:
    """
    Scores predicted extraction against gold answers.
    Handles: exact match, fuzzy string (rapidfuzz), numeric ±tolerance,
             date normalization, list matching, nested dict recursion.
    """
    def __init__(self, threshold: float = 0.85, numeric_tolerance: float = 0.02):
        self.threshold = threshold
        self.numeric_tolerance = numeric_tolerance

    def _normalize_date(self, date_str: str) -> str:
        try:
            return parse_date(date_str, fuzzy=True).strftime("%Y-%m-%d")
        except Exception:
            return str(date_str)

    def _clean_number(self, num_val: Any) -> float:
        try:
            if isinstance(num_val, (int, float)):
                return float(num_val)
            cleaned = re.sub(r'[^\d.-]', '', str(num_val))
            return float(cleaned)
        except Exception:
            return float('nan')

    def _score_field(self, pred_val: Any, gold_val: Any, field_type: str = "") -> float:
        """
        String: rapidfuzz.fuzz.ratio / 100, pass if >= threshold
        Number: 1.0 if abs(pred-gold)/gold <= tolerance else 0.0
        Date: normalize to YYYY-MM-DD, then exact match
        List: order-insensitive, average item-wise fuzzy scores
        Dict: recursive score()
        """
        if gold_val is None:
            return 1.0 if pred_val is None else 0.0
        if pred_val is None:
            return 0.0

        if isinstance(gold_val, dict):
            if not isinstance(pred_val, dict):
                return 0.0
            if not gold_val:
                return 1.0 if not pred_val else 0.0
            scores = [self._score_field(pred_val.get(k), v) for k, v in gold_val.items()]
            return sum(scores) / len(scores)

        if isinstance(gold_val, list):
            if not isinstance(pred_val, list):
                return 0.0
            if not gold_val:
                return 1.0 if not pred_val else 0.0
            
            # Simple order-insensitive matching using a greedy approach
            pred_used = set()
            total_score = 0.0
            for g_item in gold_val:
                best_score = 0.0
                best_idx = -1
                for i, p_item in enumerate(pred_val):
                    if i in pred_used:
                        continue
                    s = self._score_field(p_item, g_item)
                    if s > best_score:
                        best_score = s
                        best_idx = i
                total_score += best_score
                if best_idx != -1:
                    pred_used.add(best_idx)
            return total_score / len(gold_val)

        # Date heuristic
        if isinstance(gold_val, str) and len(gold_val) <= 15 and re.match(r'.*\d{2,4}.*', gold_val):
            # Attempt date match
            g_date = self._normalize_date(gold_val)
            p_date = self._normalize_date(str(pred_val))
            if g_date != gold_val and g_date == p_date:
                return 1.0
            
        # Number matching
        if isinstance(gold_val, (int, float)) or (isinstance(gold_val, str) and re.match(r'^[\d.,]+$', gold_val)):
            g_num = self._clean_number(gold_val)
            p_num = self._clean_number(pred_val)
            if g_num != g_num: # NaN check
                pass # fallback to string
            elif g_num == 0:
                return 1.0 if p_num == 0 else 0.0
            else:
                if p_num == p_num and abs(g_num - p_num) / abs(g_num) <= self.numeric_tolerance:
                    return 1.0
                return 0.0

        # String matching
        g_str = str(gold_val).strip().lower()
        p_str = str(pred_val).strip().lower()
        
        if not g_str:
            return 1.0 if not p_str else 0.0
            
        ratio = fuzz.ratio(g_str, p_str) / 100.0
        if ratio >= self.threshold:
            return 1.0
        return ratio # return partial credit instead of 0 for soft scoring

    def field_level_report(self, predicted: dict, gold: dict) -> dict:
        """Returns {field_name: score} for debugging/demo display"""
        report = {}
        for k, v in gold.items():
            report[k] = self._score_field(predicted.get(k), v)
        return report

    def score(self, predicted: dict, gold: dict, schema: dict) -> float:
        """Returns 0.0–1.0. Average of per-field scores."""
        if not gold:
            return 0.0
            
        report = self.field_level_report(predicted, gold)
        return sum(report.values()) / len(report)

    def __call__(self, predicted: dict, gold: dict, schema: dict) -> float:
        return self.score(predicted, gold, schema)
