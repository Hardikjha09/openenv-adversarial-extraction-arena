"""
Evaluation Harness to evaluate checkpoints on holdout sets and generate metrics for plotting.
"""

import os
import json
import random
from evaluation.elo import EloRater
from env.extraction_env import AdversarialExtractionEnv, EnvAction
from env.models import AdversaryAction, ExtractorAction, AdversaryEdit

def simulate_eval_run(num_episodes: int = 100):
    print(f"Running Evaluation Harness over {num_episodes} episodes...")
    env = AdversarialExtractionEnv(split="holdout")
    elo = EloRater()
    
    results = []
    
    for i in range(num_episodes):
        obs = env.reset()
        # Mock Adversary Action (Random policy baseline)
        if random.random() < 0.5:
            # Add simple ocr noise edit
            edit = AdversaryEdit(edit_type="ocr_noise", params={"intensity": 0.2}, token_cost=10)
            adv_action = EnvAction(action=AdversaryAction(edits=[edit], total_token_cost=10))
        else:
            adv_action = EnvAction(action=AdversaryAction(edits=[], total_token_cost=0))
            
        env.step(adv_action)
        
        # Mock Extractor Action
        doc_type = env.state.doc_type
        # In a real eval, we run the trained model on the perturbed document
        # Here we mock extraction
        mock_json = {"mock_field": "mock_value"} 
        
        ext_action = EnvAction(action=ExtractorAction(extracted_json=mock_json))
        final_obs = env.step(ext_action)
        
        ext_reward = final_obs.observation.reward
        adv_reward = env.state.adversary_reward
        
        ext_elo, adv_elo = elo.update(ext_reward, adv_reward)
        
        results.append({
            "episode": i,
            "extractor_reward": ext_reward,
            "adversary_reward": adv_reward,
            "extractor_elo": ext_elo,
            "adversary_elo": adv_elo,
            "edits_applied": len(env.state.applied_edits)
        })
        
    os.makedirs("data", exist_ok=True)
    with open("data/eval_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Evaluation complete. Extractor Elo: {ext_elo:.1f}, Adversary Elo: {adv_elo:.1f}")
    print("Metrics saved to data/eval_metrics.json")

if __name__ == "__main__":
    simulate_eval_run()
