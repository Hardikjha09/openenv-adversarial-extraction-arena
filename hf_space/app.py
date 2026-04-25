"""
Gradio Space app for Adversarial Structured-Extraction Arena.

This is a self-contained demo bundle intended for Hugging Face Spaces.
It uses the repo's adversary edit executor and schema-driven extraction UI.
If `data/corpus.json` is not present (common on Spaces), it falls back to
two built-in sample documents.
"""

import json
import gradio as gr

from env.adversary import AdversaryEditExecutor
from env.models import AdversaryEdit


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
        # Smoke check (can still be empty)
        _ = corpus.sample()
        return corpus
    except Exception:
        return None


executor = AdversaryEditExecutor()
corpus = _load_corpus_if_available()
fallback_docs = _fallback_docs()
fallback_idx = 0


def load_random_doc():
    global fallback_idx
    if corpus is not None:
        doc = corpus.sample()
        return doc["text"], json.dumps(doc["schema"], indent=2)

    doc = fallback_docs[fallback_idx % len(fallback_docs)]
    fallback_idx += 1
    return doc["text"], json.dumps(doc["schema"], indent=2)


def apply_perturbation(doc_text, schema_text, edit_intensity):
    try:
        schema = json.loads(schema_text)
    except Exception:
        schema = {}

    edit = AdversaryEdit(
        edit_type="ocr_noise",
        params={"intensity": float(edit_intensity)},
        token_cost=10,  # corrected by model validator
    )
    mod_doc, mod_schema = executor.apply_edits(doc_text, schema, [edit])

    return mod_doc, json.dumps(mod_schema, indent=2)


def extract_data(doc_text, schema_text):
    """
    Placeholder extractor.

    For a fully aligned repo demo, replace this function with a call to:
    - your hosted extractor model (HF Inference), or
    - a local model in the Space (GPU Space).
    """
    try:
        schema = json.loads(schema_text)
        extracted = {}
        if isinstance(schema, dict) and "properties" in schema and isinstance(schema["properties"], dict):
            for k in schema["properties"].keys():
                extracted[k] = "[Extracted Value]"
        return json.dumps(extracted, indent=2)
    except Exception:
        return "{}"


with gr.Blocks(title="Adversarial Extraction Arena") as demo:
    gr.Markdown("# Adversarial Structured-Extraction Arena")
    gr.Markdown("Agent A perturbs documents. Agent E extracts structured data despite the noise.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Original Document")
            doc_input = gr.TextArea(label="Document Text", lines=10)
            schema_input = gr.TextArea(label="Target Schema", lines=10)
            load_btn = gr.Button("Load Random Document")

        with gr.Column():
            gr.Markdown("### Adversary (Agent A)")
            intensity_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.2, step=0.1, label="Noise Intensity"
            )
            perturb_btn = gr.Button("Apply Perturbation")
            mod_doc_output = gr.TextArea(label="Perturbed Document", lines=10)

        with gr.Column():
            gr.Markdown("### Extractor (Agent E)")
            extract_btn = gr.Button("Run Extractor")
            extracted_output = gr.TextArea(label="Extracted JSON", lines=10)

    load_btn.click(fn=load_random_doc, inputs=[], outputs=[doc_input, schema_input])
    perturb_btn.click(
        fn=apply_perturbation,
        inputs=[doc_input, schema_input, intensity_slider],
        outputs=[mod_doc_output, schema_input],
    )
    extract_btn.click(fn=extract_data, inputs=[mod_doc_output, schema_input], outputs=[extracted_output])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

