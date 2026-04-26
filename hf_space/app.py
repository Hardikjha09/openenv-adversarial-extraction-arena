"""
Gradio Space app for Adversarial Structured-Extraction Arena.

Uses Hub LoRAs (`extractor-aea`, `adversary-aea`) when CUDA is available (enable GPU in Space settings).
Otherwise falls back to manual OCR perturbation and a placeholder extractor JSON.
"""

import json
import os

import gradio as gr

from env.adversary import AdversaryEditExecutor
from env.models import AdversaryEdit
import model_backend as mb


def _fallback_docs():
    docs = [
        {
            "text": (
                "TAX INVOICE\n"
                "Supplier: ABC Traders\n"
                "GSTIN: 29ABCDE1234F1Z5\n"
                "Invoice No: INV-1029\n"
                "Invoice Date: 12/03/2025\n"
                "Bill To: Rahul Sharma\n"
                "Phone: 9876543210\n"
                "Total Amount: ₹ 12,450.00\n"
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "supplier_name": {"type": "string"},
                    "gstin": {"type": "string"},
                    "invoice_number": {"type": "string"},
                    "invoice_date": {"type": "string"},
                    "customer_name": {"type": "string"},
                    "phone": {"type": "string"},
                    "total_amount": {"type": "number"},
                },
                "required": ["gstin", "invoice_number", "invoice_date", "total_amount"],
                "additionalProperties": False,
            },
        },
        {
            "text": (
                "BANK STATEMENT\n"
                "Account Holder: Priya Verma\n"
                "Account No: 001234567890\n"
                "IFSC: HDFC0001234\n"
                "Period: 01/01/2025 - 31/01/2025\n"
                "Closing Balance: INR 54,210.75\n"
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "account_holder": {"type": "string"},
                    "account_number": {"type": "string"},
                    "ifsc": {"type": "string"},
                    "period_start": {"type": "string"},
                    "period_end": {"type": "string"},
                    "closing_balance": {"type": "number"},
                },
                "required": ["account_number", "ifsc", "closing_balance"],
                "additionalProperties": False,
            },
        },
    ]
    return docs


def _load_corpus_if_available():
    try:
        from data.corpus import DocumentCorpus

        corpus = DocumentCorpus(split="holdout")
        _ = corpus.sample()
        return corpus
    except Exception:
        return None


executor = AdversaryEditExecutor()
corpus = _load_corpus_if_available()
fallback_docs = _fallback_docs()
fallback_idx = 0

SPACE_USE_MODELS = os.environ.get("SPACE_USE_MODELS", "1") != "0"


def refresh_status():
    if not SPACE_USE_MODELS:
        return "Trained models disabled (`SPACE_USE_MODELS=0`). Using manual / placeholder paths only."
    return mb.backend_status_message()


def load_random_doc():
    global fallback_idx
    if corpus is not None:
        doc = corpus.sample()
        return doc["text"], json.dumps(doc["schema"], indent=2), refresh_status()

    doc = fallback_docs[fallback_idx % len(fallback_docs)]
    fallback_idx += 1
    return doc["text"], json.dumps(doc["schema"], indent=2), refresh_status()


def apply_perturbation(doc_text, schema_text, intensity_slider, adversary_mode):
    try:
        schema = json.loads(schema_text)
    except Exception:
        schema = {}

    use_learned = (
        SPACE_USE_MODELS
        and adversary_mode == "Learned adversary (LoRA on GPU)"
    )

    if use_learned:
        edits = mb.run_adversary(doc_text, schema)
        if not edits:
            note = (
                "[Fell back to manual OCR: adversary LoRA unavailable — check GPU, HF_TOKEN, or logs.]\n\n"
            )
            edit = AdversaryEdit(
                edit_type="ocr_noise",
                params={"intensity": float(intensity_slider)},
                token_cost=10,
            )
            mod_doc, mod_schema = executor.apply_edits(doc_text, schema, [edit])
            return note + mod_doc, json.dumps(mod_schema, indent=2), refresh_status()
        mod_doc, mod_schema = executor.apply_edits(doc_text, schema, edits)
        return mod_doc, json.dumps(mod_schema, indent=2), refresh_status()

    edit = AdversaryEdit(
        edit_type="ocr_noise",
        params={"intensity": float(intensity_slider)},
        token_cost=10,
    )
    mod_doc, mod_schema = executor.apply_edits(doc_text, schema, [edit])
    return mod_doc, json.dumps(mod_schema, indent=2), refresh_status()


def extract_data(doc_text, schema_text, extract_mode):
    try:
        schema = json.loads(schema_text)
    except Exception:
        schema = {}

    use_model = SPACE_USE_MODELS and extract_mode == "Trained extractor (LoRA on GPU)"

    if use_model:
        extracted = mb.run_extractor(doc_text, schema)
        if extracted:
            return json.dumps(extracted, indent=2), refresh_status()
        # fall through to placeholder if model missing or parse failed

    extracted = {}
    if isinstance(schema, dict) and "properties" in schema and isinstance(schema["properties"], dict):
        for k in schema["properties"].keys():
            extracted[k] = "[Extracted Value — enable GPU + LoRA for real extraction]"
    return json.dumps(extracted, indent=2), refresh_status()


with gr.Blocks(title="Extraction Arena") as demo:
    gr.Markdown("# Adversarial Structured-Extraction Arena")
    gr.Markdown(
        "**GPU Space:** loads `HardikJha/extractor-aea` and `HardikJha/adversary-aea` (4-bit, one at a time). "
        "**CPU Space:** use manual OCR + placeholder extract, or run the [Colab notebook](https://colab.research.google.com/github/Hardikjha09/openenv-adversarial-extraction-arena/blob/main/notebooks/Train_Extractor_Colab.ipynb)."
    )

    status = gr.Textbox(label="Backend status", value=refresh_status(), lines=2)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Document & schema")
            doc_input = gr.TextArea(label="Document Text", lines=10)
            schema_input = gr.TextArea(label="Target Schema", lines=10)
            load_btn = gr.Button("Load random document")

        with gr.Column():
            gr.Markdown("### Adversary (Agent A)")
            adversary_mode = gr.Radio(
                choices=[
                    "Manual OCR noise",
                    "Learned adversary (LoRA on GPU)",
                ],
                value="Manual OCR noise",
                label="Perturbation mode",
            )
            intensity_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.2, step=0.1, label="OCR noise intensity (manual mode)"
            )
            perturb_btn = gr.Button("Apply perturbation")
            mod_doc_output = gr.TextArea(label="Perturbed document", lines=10)

        with gr.Column():
            gr.Markdown("### Extractor (Agent E)")
            extract_mode = gr.Radio(
                choices=[
                    "Placeholder (no GPU)",
                    "Trained extractor (LoRA on GPU)",
                ],
                value="Trained extractor (LoRA on GPU)",
                label="Extraction mode",
            )
            extract_btn = gr.Button("Run extractor")
            extracted_output = gr.TextArea(label="Extracted JSON", lines=10)

    load_btn.click(fn=load_random_doc, inputs=[], outputs=[doc_input, schema_input, status])
    perturb_btn.click(
        fn=apply_perturbation,
        inputs=[doc_input, schema_input, intensity_slider, adversary_mode],
        outputs=[mod_doc_output, schema_input, status],
    )
    extract_btn.click(
        fn=extract_data,
        inputs=[mod_doc_output, schema_input, extract_mode],
        outputs=[extracted_output, status],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
