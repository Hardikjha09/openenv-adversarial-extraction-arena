"""
Plot generator for training metrics and Elo ratings.
"""

import os
import json
import matplotlib.pyplot as plt

def generate_elo_plot(metrics_path: str = "data/eval_metrics.json", output_path: str = "plots/elo_ratings.png"):
    if not os.path.exists(metrics_path):
        print(f"Metrics file {metrics_path} not found.")
        return
        
    with open(metrics_path, "r") as f:
        data = json.load(f)
        
    episodes = [d["episode"] for d in data]
    ext_elo = [d["extractor_elo"] for d in data]
    adv_elo = [d["adversary_elo"] for d in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, ext_elo, label="Extractor Elo", color="blue", linewidth=2)
    plt.plot(episodes, adv_elo, label="Adversary Elo", color="red", linewidth=2)
    
    plt.title("Adversarial Co-evolution: Elo Ratings over Episodes", fontsize=14)
    plt.xlabel("Evaluation Episode", fontsize=12)
    plt.ylabel("Elo Rating", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Saved Elo plot to {output_path}")

def generate_reward_plot(metrics_path: str = "data/eval_metrics.json", output_path: str = "plots/rewards.png"):
    if not os.path.exists(metrics_path):
        return
        
    with open(metrics_path, "r") as f:
        data = json.load(f)
        
    # Apply moving average
    window = 10
    ext_reward = [d["extractor_reward"] for d in data]
    
    def moving_avg(x, w):
        return [sum(x[max(0, i-w):i+1])/len(x[max(0, i-w):i+1]) for i in range(len(x))]
        
    ext_smooth = moving_avg(ext_reward, window)
    episodes = [d["episode"] for d in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, ext_smooth, label=f"Extractor Reward (MA {window})", color="green", linewidth=2)
    plt.scatter(episodes, ext_reward, alpha=0.2, color="green", s=10)
    
    plt.title("Extractor Reward over Episodes", fontsize=14)
    plt.xlabel("Evaluation Episode", fontsize=12)
    plt.ylabel("Reward [0, 1]", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    print(f"Saved Reward plot to {output_path}")

if __name__ == "__main__":
    print("Generating plots from evaluation metrics...")
    generate_elo_plot()
    generate_reward_plot()
