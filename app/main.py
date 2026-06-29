import logging
import json
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, List, Union
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings

# Import preprocessing helpers
from src.preprocessing import preprocess_features

# Structured configuration management
class Settings(BaseSettings):
    hf_repo_id: str | None = None
    hf_token: str | None = None
    local_model_path: Path = Path("output/model_outputs/model.pkl")
    local_features_path: Path = Path("output/model_outputs/feature_list.json")
    production_log_path: Path = Path("output/production_predictions.csv")
    secondary_lookup_path: Path = Path("output/processed_data/secondary_features_lookup.csv")
    log_level: str = "INFO"

    model_config = ConfigDict(env_file=".env", extra="ignore")

settings = Settings()

# Configure logging standard
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("credit-scoring-api")

# Global variables for model, features, and lookup
model = None
expected_features = None
secondary_lookup = None

def load_model():
    global model, expected_features, secondary_lookup
    
    hf_repo_id = settings.hf_repo_id
    hf_token = settings.hf_token
    
    local_model_path = settings.local_model_path
    local_features_path = settings.local_features_path
    
    model_loaded = False
    
    if hf_repo_id:
        logger.info(f"Loading model from Hugging Face repository: {hf_repo_id}...")
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
            logger.info("Successfully loaded model and feature list from Hugging Face Hub.")
        except Exception as e:
            logger.error(f"Error loading model from Hugging Face: {e}")
            logger.info("Attempting local fallback...")
            
    if not model_loaded:
        logger.info("Loading model from local files...")
        if local_model_path.exists() and local_features_path.exists():
            try:
                with open(local_model_path, "rb") as f:
                    model = pickle.load(f)
                with open(local_features_path, "r") as f:
                    expected_features = json.load(f)
                logger.info("Successfully loaded local model and feature list.")
            except Exception as e:
                logger.error(f"Error loading local model files: {e}")
                raise RuntimeError("Failed to load model from local files.")
        else:
            logger.warning(f"Local files not found: {local_model_path} or {local_features_path}")
            logger.warning("Server starting without a model. Predictions will return 503.")

    # Load secondary features lookup table if it exists
    try:
        if settings.secondary_lookup_path.exists():
            logger.info(f"Loading secondary features lookup table from {settings.secondary_lookup_path}...")
            secondary_lookup = pd.read_csv(settings.secondary_lookup_path)
            secondary_lookup.set_index('SK_ID_CURR', inplace=True)
            logger.info(f"Loaded lookup data for {len(secondary_lookup):,} clients.")
        else:
            logger.warning(f"Lookup file not found at {settings.secondary_lookup_path}. Requests will only use application features.")
    except Exception as e:
        logger.warning(f"Failed to load secondary features lookup: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    load_model()
    yield
    # Shutdown logic
    pass

app = FastAPI(
    title="Credit Scoring API",
    description="API for predicting credit default risk using LightGBM",
    version="1.0.0",
    lifespan=lifespan,
)

# Strict Pydantic Schema for dynamic incoming applicant profiles
class ClientData(BaseModel):
    SK_ID_CURR: int | None = None

    model_config = ConfigDict(extra="allow")

class ScoreRequest(BaseModel):
    data: Union[ClientData, List[ClientData]]

def log_prediction(results: List[Dict[str, Any]], log_path: Path):
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data frame records to append
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
        write_header = not log_path.exists()
        df_log.to_csv(log_path, mode='a', index=False, header=write_header)
        logger.info(f"Logged {len(results)} predictions to {log_path}")
    except Exception as log_err:
        logger.warning(f"Failed to log production predictions: {log_err}")

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
def score(request: ScoreRequest, background_tasks: BackgroundTasks):
    if model is None or expected_features is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Score endpoint is unavailable.",
        )
        
    # Standardize input to list of dicts
    input_data = request.data
    if isinstance(input_data, list):
        raw_data = [item.model_dump() for item in input_data]
        is_single = False
    else:
        raw_data = [input_data.model_dump()]
        is_single = True
        
    try:
        # Convert input to DataFrame
        raw_df = pd.DataFrame(raw_data)
        
        # Merge with precomputed secondary features lookup on SK_ID_CURR
        if secondary_lookup is not None and "SK_ID_CURR" in raw_df.columns:
            raw_df = raw_df.join(secondary_lookup, on='SK_ID_CURR', how='left')
        
        # Apply preprocessing
        processed_df = preprocess_features(raw_df, expected_features=expected_features)
        
        # Predict probability using ensemble average
        pred_probs = []
        if isinstance(model, dict) and ("lgbm_models" in model or "xgb_models" in model or "catboost_models" in model):
            for lgbm in model.get("lgbm_models", []):
                pred_probs.append(lgbm.predict_proba(processed_df)[:, 1])
            for xgb in model.get("xgb_models", []):
                pred_probs.append(xgb.predict_proba(processed_df)[:, 1])
            for cat in model.get("catboost_models", []):
                pred_probs.append(cat.predict_proba(processed_df)[:, 1])
            
            probabilities = np.mean(pred_probs, axis=0)
        else:
            # Fallback for single model predictions
            probabilities = model.predict_proba(processed_df)[:, 1]
        
        # Format response
        results = []
        for i, prob in enumerate(probabilities):
            sk_id = raw_data[i].get("SK_ID_CURR", None)
            decision = "Approved" if prob < 0.5 else "Rejected"
            
            results.append({
                "SK_ID_CURR": int(sk_id) if sk_id is not None else None,
                "probability": float(prob),
                "decision": decision,
            })
            
        # Log predictions in background to prevent request blockage
        background_tasks.add_task(log_prediction, results, settings.production_log_path)
            
        return results[0] if is_single else results
        
    except Exception as e:
        logger.error(f"Error processing prediction request: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Error processing prediction request: {str(e)}",
        )
