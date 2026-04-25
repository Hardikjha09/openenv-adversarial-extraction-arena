import re
import random
from typing import Tuple

# OCR confusion map — realistic character substitutions
OCR_MAP = {
    '0': 'O', 'O': '0', '1': 'l', 'l': '1', 'I': '1',
    '5': 'S', 'S': '5', '8': 'B', 'B': '8', '6': 'G',
    'rn': 'm', 'vv': 'w', 'cl': 'd',
}

class AdversaryEditExecutor:
    """Applies structured edit programs to document text and schema."""

    def apply_edits(self, document: str, schema: dict, edits: list) -> Tuple[str, dict]:
        """Apply all edits in sequence. Returns (modified_doc, modified_schema)."""
        doc, sch = document, schema.copy()
        # Ensure deep copy of properties
        if "properties" in sch:
            sch["properties"] = sch["properties"].copy()
        if "required" in sch:
            sch["required"] = sch["required"].copy()
            
        for edit in edits:
            doc, sch = self.apply_single_edit(doc, sch, edit)
        return doc, sch

    def apply_single_edit(self, doc: str, schema: dict, edit) -> Tuple[str, dict]:
        t = edit.edit_type.value
        p = edit.params
        try:
            if t == "rename_field":
                return self.rename_field(doc, schema, p.get("old_name", ""), p.get("new_name", ""))
            elif t == "swap_type":
                return self.swap_type(doc, schema, p.get("field", ""), p.get("new_type", "string"))
            elif t == "inject_distractor":
                return self.inject_distractor(doc, schema, p.get("content", ""))
            elif t == "mutate_format":
                return self.mutate_format(doc, schema, p.get("field", ""), p.get("pattern", ""))
            elif t == "add_required_field":
                return self.add_required_field(doc, schema, p.get("name", ""), p.get("value", ""))
            elif t == "ocr_noise":
                return self.ocr_noise(doc, schema, float(p.get("intensity", 0.3)))
            elif t == "swap_columns":
                return self.swap_columns(doc, schema, p.get("col_a", 0), p.get("col_b", 1))
        except Exception as e:
            # Silently fail on bad adversary actions
            pass
        return doc, schema

    def rename_field(self, doc: str, schema: dict, old_name: str, new_name: str) -> Tuple[str, dict]:
        if not old_name or not new_name:
            return doc, schema
        # Replace old_name with new_name in doc text (case-insensitive, word-boundary safe)
        pattern = re.compile(r'\b' + re.escape(old_name) + r'\b', re.IGNORECASE)
        new_doc = pattern.sub(new_name, doc)
        
        # Also update schema: rename the property key
        new_schema = schema
        if "properties" in new_schema and old_name in new_schema["properties"]:
            prop = new_schema["properties"].pop(old_name)
            new_schema["properties"][new_name] = prop
            
        if "required" in new_schema and old_name in new_schema["required"]:
            new_schema["required"].remove(old_name)
            new_schema["required"].append(new_name)
            
        return new_doc, new_schema
        
    def swap_type(self, doc: str, schema: dict, field: str, new_type: str) -> Tuple[str, dict]:
        if "properties" in schema and field in schema["properties"]:
            schema["properties"][field]["type"] = new_type
        return doc, schema

    def ocr_noise(self, doc: str, schema: dict, intensity: float) -> Tuple[str, dict]:
        # Apply OCR_MAP substitutions to `intensity` fraction of eligible chars
        intensity = max(0.0, min(1.0, intensity))
        chars = list(doc)
        for i, char in enumerate(chars):
            if random.random() < intensity:
                if char in OCR_MAP:
                    chars[i] = OCR_MAP[char]
                elif char.isdigit() and random.random() < 0.1:
                    chars[i] = char + ' ' if random.random() > 0.5 else ''
        return "".join(chars), schema

    def mutate_format(self, doc: str, schema: dict, field: str, pattern: str) -> Tuple[str, dict]:
        if pattern == "date_dmy_to_mdy":
            doc = re.sub(r'(\d{2})/(\d{2})/(\d{4})', r'\2-\1-\3', doc)
        elif pattern == "date_dmy_to_iso":
            doc = re.sub(r'(\d{2})/(\d{2})/(\d{4})', r'\3-\2-\1', doc)
        elif pattern == "currency_symbol_to_text":
            doc = doc.replace('₹', 'INR ')
        elif pattern == "phone_compact_to_dashed":
            doc = re.sub(r'(?<!\d)(\d{10})(?!\d)', r'\g<1>'[:3] + '-' + r'\g<1>'[3:6] + '-' + r'\g<1>'[6:], doc)
        return doc, schema

    def inject_distractor(self, doc: str, schema: dict, content: str) -> Tuple[str, dict]:
        if not content:
            content = "Random distractor line item that means nothing."
        lines = doc.split("\n")
        # Insert near the end but not at the very end
        idx = max(0, len(lines) - 2 - random.randint(0, 3))
        lines.insert(idx, content)
        return "\n".join(lines), schema

    def add_required_field(self, doc: str, schema: dict, name: str, value: str) -> Tuple[str, dict]:
        if not name:
            return doc, schema
        # Append "name: value" to document
        doc += f"\n{name}: {value}"
        # AND add name to schema required fields
        if "properties" in schema:
            schema["properties"][name] = {"type": "string"}
        if "required" in schema:
            schema["required"].append(name)
        return doc, schema

    def swap_columns(self, doc: str, schema: dict, col_a: int, col_b: int) -> Tuple[str, dict]:
        # Too complex for quick implementation, just return doc for now
        # Would parse tables and swap
        return doc, schema

    def is_document_parseable(self, original_doc: str, modified_doc: str) -> bool:
        # Fail if more than 40% of key:value patterns from original are unrecognizable
        orig_matches = len(re.findall(r"[\w\s]+:\s*[\w\s₹/\-\.,]+", original_doc))
        mod_matches = len(re.findall(r"[\w\s]+:\s*[\w\s₹/\-\.,]+", modified_doc))
        
        if orig_matches == 0:
            return True # Not applicable
            
        return (mod_matches / orig_matches) >= 0.6

    def validate_budget(self, edits: list, budget: int) -> bool:
        return sum(e.token_cost for e in edits) <= budget
