import os
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Union
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import preprocessing helpers
from src.preprocessing import preprocess_features

app = FastAPI(
    title="Credit Scoring API...",
    description="API for predicting credit default risk using LightGBM",
    version="1.0.0",
)

# Global variables for model and features
model = None
expected_features = None

class ScoreRequest(BaseModel):
    # Can be a single client dictionary or a list of client dictionaries
    data: Union[Dict[str, Any], List[Dict[str, Any]]]

@app.on_event("startup")
def load_model():
    global model, expected_features
    
    # Check for Hugging Face configuration
    hf_repo_id = os.environ.get("HF_REPO_ID")
    hf_token = os.environ.get("HF_TOKEN")
    
    # Local fallback paths
    local_model_path = Path("output/model_outputs/model.pkl")
    local_features_path = Path("output/model_outputs/feature_list.json")
    
    model_loaded = False
    
    if hf_repo_id:
        print(f"Loading model from Hugging Face repository: {hf_repo_id}...")
        try:
            from huggingface_hub import hf_hub_download
            
            # Download model.pkl
            hf_model_path = hf_hub_download(
                repo_id=hf_repo_id,
                filename="model.pkl",
                token=hf_token,
            )
            # Download feature_list.json
            hf_features_path = hf_hub_download(
                repo_id=hf_repo_id,
                filename="feature_list.json",
                token=hf_token,
            )
            
            with open(hf_model_path, "rb") as f:
                model = pickle.load(f)
            with open(hf_features_path, "r") as f:
                expected_features = json.load(f)
                
            model_loaded = True
            print("Successfully loaded model and feature list from Hugging Face Hub.")
        except Exception as e:
            print(f"Error loading model from Hugging Face: {e}")
            print("Attempting local fallback...")
            
    if not model_loaded:
        print("Loading model from local files...")
        if local_model_path.exists() and local_features_path.exists():
            try:
                with open(local_model_path, "rb") as f:
                    model = pickle.load(f)
                with open(local_features_path, "r") as f:
                    expected_features = json.load(f)
                print("Successfully loaded local model and feature list.")
            except Exception as e:
                print(f"Error loading local model files: {e}")
                raise RuntimeError("Failed to load model from local files.")
        else:
            print(f"Local files not found: {local_model_path} or {local_features_path}")
            print("Server starting without a model. Predictions will return 503.")

@app.get("/health")
def health():
    if model is None or expected_features is None:
        return {
            "status": "unhealthy",
            "message": "Model is not loaded. Please train the model or verify Hugging Face configuration.",
        }
    return {
        "status": "healthy",
        "model_loaded": True,
        "features_count": len(expected_features),
    }

@app.post("/score")
def score(request: ScoreRequest):
    if model is None or expected_features is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Score endpoint is unavailable.",
        )
        
    # Standardize input to list of dicts
    input_data = request.data
    if isinstance(input_data, dict):
        input_data = [input_data]
        is_single = True
    else:
        is_single = False
        
    try:
        # Convert input to DataFrame
        raw_df = pd.DataFrame(input_data)
        
        # Apply preprocessing
        processed_df = preprocess_features(raw_df, expected_features=expected_features)
        
        # Predict probability of default (class 1)
        probabilities = model.predict_proba(processed_df)[:, 1]
        
        # Format response
        results = []
        for i, prob in enumerate(probabilities):
            # Check if identifier exists, otherwise use index
            sk_id = input_data[i].get("SK_ID_CURR", None)
            
            # Decide threshold (standard 0.5, or a credit scoring target e.g. 0.3/0.1)
            # Home credit default rate is around 8%, so a low threshold is often better. Let's make it 0.5 for standard default.
            decision = "Approved" if prob < 0.5 else "Rejected"
            
            results.append({
                "SK_ID_CURR": int(sk_id) if sk_id is not None else None,
                "probability": float(prob),
                "decision": decision,
            })
            
        # Log predictions to a file for drift monitoring
        try:
            prod_log_path = Path("output/production_predictions.csv")
            prod_log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare df to append
            log_data = []
            import datetime
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            
            for res in results:
                log_data.append({
                    "timestamp": timestamp,
                    "SK_ID_CURR": res["SK_ID_CURR"],
                    "probability": res["probability"],
                    "decision": res["decision"]
                })
            df_log = pd.DataFrame(log_data)
            
            # Write header if file does not exist
            write_header = not prod_log_path.exists()
            df_log.to_csv(prod_log_path, mode='a', index=False, header=write_header)
        except Exception as log_err:
            print(f"Warning: Failed to log production predictions: {log_err}")
            
        return results[0] if is_single else results
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing prediction request: {str(e)}",
        )
