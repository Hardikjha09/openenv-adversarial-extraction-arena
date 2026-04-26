"""
Gradio Space app for Adversarial Structured-Extraction Arena.

Uses Hub LoRAs (`extractor-aea`, `adversary-aea`) when CUDA is available (enable GPU in Space settings).
Otherwise falls back to manual OCR perturbation and a placeholder extractor JSON.
"""

import json
import os
import random

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
        {
            "text": (
                "DELIVERY CHALLAN\n"
                "DC No: DC-77821\n"
                "Date: 18/04/2025\n"
                "Ship From: Pune WH-02\n"
                "Ship To: Neha Kapoor, Bengaluru 560001\n"
                "LR Number: MH14AB9922\n"
                "Items: LED Panels — Qty 24\n"
                "Vehicle: MH 12 GH 4411\n"
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "dc_number": {"type": "string"},
                    "dc_date": {"type": "string"},
                    "consignee_name": {"type": "string"},
                    "consignee_city_pin": {"type": "string"},
                    "lr_number": {"type": "string"},
                    "quantity": {"type": "number"},
                    "vehicle_reg": {"type": "string"},
                },
                "required": ["dc_number", "dc_date", "lr_number"],
                "additionalProperties": False,
            },
        },
        {
            "text": (
                "PURCHASE ORDER\n"
                "PO Ref: PO-MUM-2025-0091\n"
                "Vendor: Laxmi Steels\n"
                "GSTIN Vendor: 27AABCL1234C1Z8\n"
                "Delivery Site: Navi Mumbai Plant\n"
                "Line Total: ₹ 3,88,920.00\n"
                "IGST 18%: ₹ 70,005.60\n"
                "Grand Total: ₹ 4,58,925.60\n"
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "po_reference": {"type": "string"},
                    "vendor_name": {"type": "string"},
                    "vendor_gstin": {"type": "string"},
                    "delivery_site": {"type": "string"},
                    "line_total": {"type": "number"},
                    "igst_amount": {"type": "number"},
                    "grand_total": {"type": "number"},
                },
                "required": ["po_reference", "vendor_gstin", "grand_total"],
                "additionalProperties": False,
            },
        },
        {
            "text": (
                "RENT RECEIPT\n"
                "Receipt No: RNT-4401\n"
                "Landlord: Shri Venkatesh Iyer\n"
                "Tenant: Arjun Mehta\n"
                "Property: Flat 402, Tower B, Whitefield\n"
                "Month: March 2025\n"
                "Rent: ₹ 28,000\n"
                "Mode: UPI ref 8877665544332211\n"
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "receipt_number": {"type": "string"},
                    "landlord": {"type": "string"},
                    "tenant": {"type": "string"},
                    "property_summary": {"type": "string"},
                    "period_month": {"type": "string"},
                    "rent_amount": {"type": "number"},
                    "payment_ref": {"type": "string"},
                },
                "required": ["receipt_number", "tenant", "rent_amount"],
                "additionalProperties": False,
            },
        },
        {
            "text": (
                "SALARY SLIP — MARCH 2025\n"
                "Employee: Kavya Nair\n"
                "Employee ID: EMP-88331\n"
                "PAN: ABCPN1234D\n"
                "UAN: 101234567890\n"
                "Basic: ₹ 65,000\n"
                "HRA: ₹ 26,000\n"
                "Deduction PF: ₹ 7,800\n"
                "Net Pay: ₹ 92,400\n"
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "employee_name": {"type": "string"},
                    "employee_id": {"type": "string"},
                    "pan": {"type": "string"},
                    "uan": {"type": "string"},
                    "basic_salary": {"type": "number"},
                    "hra": {"type": "number"},
                    "pf_deduction": {"type": "number"},
                    "net_pay": {"type": "number"},
                },
                "required": ["employee_id", "pan", "net_pay"],
                "additionalProperties": False,
            },
        },
        {
            "text": (
                "ELECTRICITY BILL — BESCOM\n"
                "Consumer: Sunita Rao\n"
                "RR Number: RR-BLR-00998877\n"
                "Billing Period: 15/02/2025 to 14/03/2025\n"
                "Units Consumed: 412 kWh\n"
                "Amount Payable: ₹ 3,842.00\n"
                "Due Date: 05/04/2025\n"
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "consumer_name": {"type": "string"},
                    "rr_number": {"type": "string"},
                    "billing_start": {"type": "string"},
                    "billing_end": {"type": "string"},
                    "units_kwh": {"type": "number"},
                    "amount_payable": {"type": "number"},
                    "due_date": {"type": "string"},
                },
                "required": ["rr_number", "units_kwh", "amount_payable"],
                "additionalProperties": False,
            },
        },
        {
            "text": (
                "MOTOR INSURANCE — RENEWAL QUOTE\n"
                "Policy Type: Comprehensive\n"
                "Insured: Rohit Desai\n"
                "Vehicle: KA03MM7722 (Hyundai Creta)\n"
                "IDV: ₹ 9,85,000\n"
                "NCB: 35%\n"
                "Premium: ₹ 14,260\n"
                "Quote Valid Till: 30/04/2025\n"
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "policy_type": {"type": "string"},
                    "insured_name": {"type": "string"},
                    "vehicle_reg": {"type": "string"},
                    "vehicle_model": {"type": "string"},
                    "idv": {"type": "number"},
                    "ncb_percent": {"type": "number"},
                    "premium": {"type": "number"},
                    "quote_valid_until": {"type": "string"},
                },
                "required": ["insured_name", "vehicle_reg", "premium"],
                "additionalProperties": False,
            },
        },
        {
            "text": (
                "FIXED DEPOSIT ADVICE\n"
                "Bank: ICICI Bank\n"
                "FD Account: FD/882211003344\n"
                "Customer: Meera Joshi\n"
                "Principal: ₹ 5,00,000\n"
                "Tenure: 390 days\n"
                "Interest Rate: 7.25% p.a.\n"
                "Maturity Value: ₹ 5,38,900 (approx)\n"
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "bank_name": {"type": "string"},
                    "fd_account": {"type": "string"},
                    "customer_name": {"type": "string"},
                    "principal": {"type": "number"},
                    "tenure_days": {"type": "number"},
                    "interest_rate_percent": {"type": "number"},
                    "maturity_value": {"type": "number"},
                },
                "required": ["fd_account", "principal", "interest_rate_percent"],
                "additionalProperties": False,
            },
        },
        {
            "text": (
                "OUTPATIENT BILL\n"
                "Hospital: Apollo Clinic Indiranagar\n"
                "Patient: Vikram Singh\n"
                "UHID: UHID-AP-992211\n"
                "Visit Date: 22/03/2025\n"
                "Consultation: ₹ 850\n"
                "Lab Tests: ₹ 2,400\n"
                "Pharmacy: ₹ 1,175\n"
                "Total: ₹ 4,425\n"
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "hospital_name": {"type": "string"},
                    "patient_name": {"type": "string"},
                    "uhid": {"type": "string"},
                    "visit_date": {"type": "string"},
                    "consultation_fee": {"type": "number"},
                    "lab_charges": {"type": "number"},
                    "pharmacy_charges": {"type": "number"},
                    "total_amount": {"type": "number"},
                },
                "required": ["uhid", "visit_date", "total_amount"],
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

SPACE_USE_MODELS = os.environ.get("SPACE_USE_MODELS", "1") != "0"


def refresh_status():
    if not SPACE_USE_MODELS:
        return "Trained models disabled (`SPACE_USE_MODELS=0`). Using manual / placeholder paths only."
    return mb.backend_status_message()


def load_random_doc():
    if corpus is not None:
        doc = corpus.sample()
        return doc["text"], json.dumps(doc["schema"], indent=2), refresh_status()

    doc = random.choice(fallback_docs)
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
