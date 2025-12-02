import argparse
import os
from huggingface_hub import snapshot_download

def download_llm_model(repo_id: str, target_dir: str):
    """
    Downloads an LLM model from Hugging Face Hub.

    Args:
        repo_id (str): The model ID on Hugging Face Hub (e.g., "mistralai/Mistral-7B-v0.1").
        target_dir (str): The local directory to save the model files.
    """
    print(f"[!] Downloading model '{repo_id}' to '{target_dir}'...")
    try:
        snapshot_download(repo_id=repo_id, local_dir=target_dir, local_dir_use_symlinks=False)
        print(f"[!] Successfully downloaded model '{repo_id}' to '{target_dir}'.")
    except Exception as e:
        print(f"[!] Error downloading model '{repo_id}': {e}")
        print("[!] Please ensure the repo_id is correct and you have internet access.")
        print("[!] If it's a private model, you might need to run `huggingface-cli login` first.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download an LLM model from Hugging Face Hub.")
    parser.add_argument("--repo-id", type=str, default="mistralai/Mistral-7B-v0.1", help="The model ID on Hugging Face Hub.")
    parser.add_argument("--target-dir", type=str, default="ckpt/pretrained_ckpt/llm_ckpt/Mistral-7B-v0.1", help="The local directory to save the model files.")
    args = parser.parse_args()

    download_llm_model(args.repo_id, args.target_dir)