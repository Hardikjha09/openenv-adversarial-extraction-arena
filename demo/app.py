"""
Gradio Demo for Adversarial Structured-Extraction Arena
Shows the side-by-side interaction of Adversary and Extractor.
"""

import gradio as gr
import json
import random
from data.corpus import DocumentCorpus
from env.adversary import AdversaryEditExecutor
from env.models import AdversaryEdit

corpus = DocumentCorpus(split="holdout")
executor = AdversaryEditExecutor()

def load_random_doc():
    doc = corpus.sample()
    return doc["text"], json.dumps(doc["schema"], indent=2)

def apply_perturbation(doc_text, schema_text, edit_intensity):
    try:
        schema = json.loads(schema_text)
    except:
        schema = {}
        
    edit = AdversaryEdit(edit_type="ocr_noise", params={"intensity": float(edit_intensity)}, token_cost=10)
    mod_doc, mod_schema = executor.apply_edits(doc_text, schema, [edit])
    
    return mod_doc, json.dumps(mod_schema, indent=2)

def extract_data(doc_text, schema_text):
    # Mocking extractor response for demo purposes
    # In a real demo, this would call the loaded Qwen model
    try:
        schema = json.loads(schema_text)
        extracted = {}
        if "properties" in schema:
            for k in schema["properties"]:
                extracted[k] = "[Extracted Value]"
        return json.dumps(extracted, indent=2)
    except:
        return "{}"

with gr.Blocks(title="Adversarial Extraction Arena") as demo:
    gr.Markdown("# ⚔️ Adversarial Structured-Extraction Arena")
    gr.Markdown("Agent A (Adversary) perturbs documents. Agent E (Extractor) tries to extract structured data despite the noise.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Original Document")
            doc_input = gr.TextArea(label="Document Text", lines=10)
            schema_input = gr.TextArea(label="Target Schema", lines=10)
            load_btn = gr.Button("Load Random Document")
            
        with gr.Column():
            gr.Markdown("### Adversary (Agent A)")
            intensity_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, label="Noise Intensity")
            perturb_btn = gr.Button("Apply Perturbation")
            mod_doc_output = gr.TextArea(label="Perturbed Document", lines=10)
            
        with gr.Column():
            gr.Markdown("### Extractor (Agent E)")
            extract_btn = gr.Button("Run Extractor")
            extracted_output = gr.TextArea(label="Extracted JSON", lines=10)
            
    load_btn.click(fn=load_random_doc, inputs=[], outputs=[doc_input, schema_input])
    perturb_btn.click(fn=apply_perturbation, inputs=[doc_input, schema_input, intensity_slider], outputs=[mod_doc_output, schema_input])
    extract_btn.click(fn=extract_data, inputs=[mod_doc_output, schema_input], outputs=[extracted_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
