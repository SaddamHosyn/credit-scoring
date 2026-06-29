import json
import pickle
import re
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("kaggle-submission")

TEST_FILE = Path("output/processed_data/application_test_engineered.csv")
RAW_TEST_FILE = Path("home-credit-default-risk/application_test.csv")
MODEL_FILE = Path("output/model_outputs/model.pkl")
FEATURES_FILE = Path("output/model_outputs/feature_list.json")

OUTPUT_DIR = Path("output/kaggle")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUB_FILE = OUTPUT_DIR / "submission.csv"

def clean_feature_names(columns):
    cleaned = []
    seen = {}
    for col in columns:
        new_col = re.sub(r"[^A-Za-z0-9_]+", "", str(col))
        if new_col == "":
            new_col = "feature"
        if new_col in seen:
            seen[new_col] += 1
            new_col = f"{new_col}_{seen[new_col]}"
        else:
            seen[new_col] = 0
        cleaned.append(new_col)
    return cleaned

def main():
    logger.info("=== GENERATING KAGGLE SUBMISSION FROM ENSEMBLE ===")
    
    # Verify files exist
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Missing test engineered file: {TEST_FILE}")
    if not RAW_TEST_FILE.exists():
        raise FileNotFoundError(f"Missing raw test file: {RAW_TEST_FILE}")
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Missing trained ensemble model file: {MODEL_FILE}")
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"Missing feature list file: {FEATURES_FILE}")
        
    # 1. Load models and features
    logger.info(f"Loading ensemble models from {MODEL_FILE}...")
    with open(MODEL_FILE, "rb") as f:
        model_dict = pickle.load(f)
        
    logger.info(f"Loading feature list from {FEATURES_FILE}...")
    with open(FEATURES_FILE, "r") as f:
        feature_list = json.load(f)
        
    # 2. Load test datasets
    logger.info("Loading test data...")
    test = pd.read_csv(TEST_FILE)
    raw_test = pd.read_csv(RAW_TEST_FILE)
    
    logger.info(f"Raw test size: {raw_test.shape[0]:,}")
    
    # 3. Align features
    test.columns = clean_feature_names(test.columns)
    test = test.reindex(columns=feature_list, fill_value=0)
    
    # 4. Predict probabilities using ensembled models
    logger.info("Predicting default probabilities using 15-model ensemble (5 folds x 3 models)...")
    pred_probs = []
    
    # LightGBM
    lgbm_count = len(model_dict.get("lgbm_models", []))
    logger.info(f"Predicting with {lgbm_count} LightGBM models...")
    for idx, lgbm in enumerate(model_dict.get("lgbm_models", [])):
        pred_probs.append(lgbm.predict_proba(test)[:, 1])
        
    # XGBoost
    xgb_count = len(model_dict.get("xgb_models", []))
    logger.info(f"Predicting with {xgb_count} XGBoost models...")
    for idx, xgb in enumerate(model_dict.get("xgb_models", [])):
        pred_probs.append(xgb.predict_proba(test)[:, 1])
        
    # CatBoost
    cat_count = len(model_dict.get("catboost_models", []))
    logger.info(f"Predicting with {cat_count} CatBoost models...")
    for idx, cat in enumerate(model_dict.get("catboost_models", [])):
        pred_probs.append(cat.predict_proba(test)[:, 1])
        
    # 5. Average predictions
    logger.info("Averaging ensembled predictions...")
    final_probs = np.mean(pred_probs, axis=0)
    
    # 6. Save submission file
    submission = pd.DataFrame({
        "SK_ID_CURR": raw_test["SK_ID_CURR"],
        "TARGET": final_probs
    })
    
    submission.to_csv(SUB_FILE, index=False)
    logger.info(f"Submission file saved successfully to: {SUB_FILE.resolve()}")
    logger.info("First few rows of submission:")
    print(submission.head())

if __name__ == "__main__":
    main()
