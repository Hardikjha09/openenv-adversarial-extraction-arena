"""
Hugging Face Space deployment script.
Pushes the Gradio Demo and models to HF hub.
"""

import os
from huggingface_hub import HfApi

def push_to_hub():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN environment variable not set. Please set it to push to Hugging Face.")
        return
        
    api = HfApi()
    repo_id = "your_username/adversarial-extraction-arena" # Replace with actual username
    
    print(f"Pushing to Hugging Face Hub: {repo_id}")
    
    try:
        api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
        
        # Upload demo app
        api.upload_folder(
            folder_path="demo",
            path_in_repo="",
            repo_id=repo_id,
            repo_type="space"
        )
        
        print(f"Successfully pushed to https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"Failed to push to Hub: {e}")

if __name__ == "__main__":
    push_to_hub()
