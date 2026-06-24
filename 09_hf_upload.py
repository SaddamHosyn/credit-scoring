import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def main():
    parser = argparse.ArgumentParser(description="Upload Credit Scoring Model to Hugging Face Hub")
    parser.add_argument(
        "--repo_id",
        type=str,
        default=os.environ.get("HF_REPO_ID"),
        help="Hugging Face repository ID (e.g., 'username/credit-scoring-lgbm')",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face API token",
    )
    
    args = parser.parse_args()
    
    repo_id = args.repo_id
    token = args.token
    
    if not repo_id:
        print("Error: Hugging Face Repository ID (--repo_id or HF_REPO_ID env var) is required.")
        return
        
    if not token:
        print("Warning: No token provided. If the repository is private or requires authorization, this might fail.")

    model_dir = Path("output/model_outputs")
    model_path = model_dir / "model.pkl"
    features_path = model_dir / "feature_list.json"
    
    if not model_path.exists():
        print(f"Error: Model pickle not found at {model_path}. Did you run the training script first?")
        return
        
    if not features_path.exists():
        print(f"Error: Feature list not found at {features_path}.")
        return

    print("=" * 60)
    print("UPLOADING MODEL TO HUGGING FACE HUB")
    print(f"Repository: {repo_id}")
    print("=" * 60)
    
    api = HfApi()
    
    # 1. Create the repository on Hugging Face if it doesn't exist
    try:
        print(f"Ensuring repository '{repo_id}' exists...")
        create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True)
        print("Repository is ready.")
    except Exception as e:
        print(f"Could not create repository (or it already exists): {e}")

    # 2. Upload the model pickle file
    print(f"Uploading model.pkl...")
    try:
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo="model.pkl",
            repo_id=repo_id,
            token=token,
            repo_type="model",
        )
        print("model.pkl successfully uploaded.")
    except Exception as e:
        print(f"Error uploading model.pkl: {e}")
        return

    # 3. Upload the features JSON file
    print(f"Uploading feature_list.json...")
    try:
        api.upload_file(
            path_or_fileobj=str(features_path),
            path_in_repo="feature_list.json",
            repo_id=repo_id,
            token=token,
            repo_type="model",
        )
        print("feature_list.json successfully uploaded.")
    except Exception as e:
        print(f"Error uploading feature_list.json: {e}")
        return
        
    # 4. Create a README.md on Hugging Face (model card)
    readme_content = f"""---
language: en
license: mit
tags:
- tabular-classification
- credit-scoring
- lightgbm
- mlops
---

# Credit Scoring LightGBM Model

This model predicts the probability of default for credit applicants. 
It is trained on the Kaggle Home Credit Default Risk dataset.

## Files
- `model.pkl`: LightGBM Classifier model weights.
- `feature_list.json`: JSON file listing the precise expected columns and order for serving/inference.
"""
    
    readme_path = model_dir / "HF_README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)
        
    try:
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
            repo_type="model",
        )
        print("README.md model card successfully uploaded.")
    except Exception as e:
        print(f"Error uploading README.md: {e}")

    print("\nAll artifacts uploaded to Hugging Face successfully!")

if __name__ == "__main__":
    main()
