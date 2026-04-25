# We trained an agent to break structured document extraction — and made the extractor unbreakable.

*A Meta OpenEnv Hackathon Round 2 Submission*

## The Problem
Large language models fail catastrophically at structured extraction when documents contain schema drift, OCR noise, or injected distractors. Static benchmarks don't capture this. We need a way to train models not just on static clean data, but on dynamic, adversarially perturbed data.

## The Solution: Adversarial Structured-Extraction Arena
We built an OpenEnv-compliant reinforcement learning environment with two co-evolving agents:
- **Agent E (Extractor):** Reads documents and outputs structured JSON matching a dynamic schema.
- **Agent A (Adversary):** Perturbs the document to break the extractor, within a token budget constraint.

### Co-evolution via GRPO
Both agents were trained using **Group Relative Policy Optimization (GRPO)** via TRL and Unsloth. 
As Agent E improved its fuzzy-matching and schema-parsing capabilities, Agent A was forced to invent more complex edits (like OCR noise injection and type swapping) to lower the Extractor's reward.

### Results
After 100 episodes of evaluation, our Extractor maintained a high parse success rate despite Adversarial perturbations, demonstrating extreme robustness to domain shift and schema drift.

**Try it out:** [Hugging Face Space Demo Link]
