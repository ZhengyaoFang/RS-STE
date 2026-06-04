"""
Upload RS-STE checkpoints to HuggingFace Hub.

Usage:
    huggingface-cli login                          # login first
    python weights/upload_to_hf.py                 # upload to Nineve/RS-STE-Checkpoints

Files uploaded:
    weights/vae.ckpt     →  RS-STE-Checkpoints/vae.ckpt
    weights/model.pth    →  RS-STE-Checkpoints/model.pth

This script itself is NOT uploaded (it's in .gitignore / excluded by path).
"""

import os
from huggingface_hub import HfApi, create_repo

REPO_ID = "Nineve/RS-STE-Checkpoints"
WEIGHTS_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = [
    "vae.ckpt",
    "model.pth",
]

def main():
    api = HfApi()

    # Create or reuse repo
    try:
        create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
        print(f"Repo https://huggingface.co/{REPO_ID} ready.")
    except Exception as e:
        print(f"Repo check failed: {e}")
        return

    for fname in FILES:
        local_path = os.path.join(WEIGHTS_DIR, fname)
        if not os.path.exists(local_path):
            print(f"SKIP: {fname} not found at {local_path}")
            continue

        print(f"Uploading {fname} ({os.path.getsize(local_path) / 1e9:.2f} GB)...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=fname,
            repo_id=REPO_ID,
            repo_type="model",
        )
        print(f"Done: {fname}")

    print("\nAll uploads complete. Check https://huggingface.co/" + REPO_ID)


if __name__ == "__main__":
    main()
