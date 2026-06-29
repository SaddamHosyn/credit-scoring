import os
import gc
import json
import pickle
import re
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("model-training")

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["figure.dpi"] = 110

try:
    from lightgbm import LGBMClassifier
except ImportError:
    raise ImportError("lightgbm is not installed.")

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError("xgboost is not installed.")

try:
    from catboost import CatBoostClassifier
except ImportError:
    raise ImportError("catboost is not installed.")

INPUT_FILE = Path("output/processed_data/application_train_engineered.csv")
TARGET_COL = "TARGET"
OUTPUT_DIR = Path("output/model_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Engineered dataset not found: {INPUT_FILE}")

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
    logger.info("=== STARTING MODEL TRAINING & ENSEMBLING ===")
    
    # 1. Load data
    logger.info(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    df.columns = clean_feature_names(df.columns)
    
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")
        
    y = df[TARGET_COL].values
    X = df.drop(columns=[TARGET_COL])
    
    logger.info(f"Data shape: {X.shape[0]:,} rows x {X.shape[1]:,} features")
    
    # 2. Stratified 5-Fold Setup
    logger.info("Initializing Stratified 5-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_lgbm = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))
    oof_ensemble = np.zeros(len(X))
    
    lgbm_models = []
    xgb_models = []
    cat_models = []
    
    lgbm_params = {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    xgb_params = {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'auc'
    }
    
    cat_params = {
        'iterations': 200,
        'learning_rate': 0.05,
        'depth': 6,
        'eval_metric': 'AUC',
        'random_seed': 42,
        'thread_count': -1,
        'verbose': False
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"--- Training Fold {fold+1} ---")
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        # 1. LightGBM
        logger.info(f"Training LightGBM on Fold {fold+1}...")
        lgbm = LGBMClassifier(**lgbm_params)
        lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])
        oof_lgbm[val_idx] = lgbm.predict_proba(X_val)[:, 1]
        lgbm_models.append(lgbm)
        
        # 2. XGBoost
        logger.info(f"Training XGBoost on Fold {fold+1}...")
        xgb = XGBClassifier(**xgb_params)
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = xgb.predict_proba(X_val)[:, 1]
        xgb_models.append(xgb)
        
        # 3. CatBoost
        logger.info(f"Training CatBoost on Fold {fold+1}...")
        cat = CatBoostClassifier(**cat_params)
        cat.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        oof_cat[val_idx] = cat.predict_proba(X_val)[:, 1]
        cat_models.append(cat)
        
        # Ensemble average
        oof_ensemble[val_idx] = (oof_lgbm[val_idx] + oof_xgb[val_idx] + oof_cat[val_idx]) / 3
        
        fold_auc = roc_auc_score(y_val, oof_ensemble[val_idx])
        logger.info(f"Fold {fold+1} Ensemble ROC-AUC: {fold_auc:.4f}")
        
        del X_train, y_train, X_val, y_val
        gc.collect()
        
    auc_lgbm = roc_auc_score(y, oof_lgbm)
    auc_xgb = roc_auc_score(y, oof_xgb)
    auc_cat = roc_auc_score(y, oof_cat)
    auc_ensemble = roc_auc_score(y, oof_ensemble)
    
    logger.info(f"Overall OOF LightGBM ROC-AUC: {auc_lgbm:.4f}")
    logger.info(f"Overall OOF XGBoost ROC-AUC: {auc_xgb:.4f}")
    logger.info(f"Overall OOF CatBoost ROC-AUC: {auc_cat:.4f}")
    logger.info(f"Overall OOF Ensemble ROC-AUC: {auc_ensemble:.4f}")
    
    # Save overall metrics
    metrics_df = pd.DataFrame({
        "model": ["lightgbm", "xgboost", "catboost", "ensemble"],
        "metric": ["roc_auc", "roc_auc", "roc_auc", "roc_auc"],
        "value": [auc_lgbm, auc_xgb, auc_cat, auc_ensemble]
    })
    metrics_df.to_csv(OUTPUT_DIR / "ensemble_metrics.csv", index=False)
    
    # 3. Save model dictionary to model.pkl
    logger.info("Saving ensembled models to model.pkl...")
    model_dict = {
        "lgbm_models": lgbm_models,
        "xgb_models": xgb_models,
        "catboost_models": cat_models
    }
    with open(OUTPUT_DIR / "model.pkl", "wb") as f:
        pickle.dump(model_dict, f)
        
    # Save feature names
    logger.info("Saving feature list...")
    features = list(X.columns)
    with open(OUTPUT_DIR / "feature_list.json", "w") as f:
        json.dump(features, f)
        
    # 4. ROC curve plot
    logger.info("Generating evaluation plots...")
    fpr, tpr, _ = roc_curve(y, oof_ensemble)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f"Ensemble (AUC = {auc_ensemble:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Out-Of-Fold Ensemble")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ensemble_roc_curve.png", bbox_inches="tight")
    plt.close()
    
    # 5. Save Feature Importance (from first LightGBM model as surrogate representation)
    importances = lgbm_models[0].feature_importances_
    fi_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    }).sort_values("importance", ascending=False)
    
    fi_df.to_csv(OUTPUT_DIR / "ensemble_feature_importances.csv", index=False)
    
    top_n = 20
    plt.figure(figsize=(10, 6))
    sns.barplot(data=fi_df.head(top_n), x="importance", y="feature", palette="viridis")
    plt.title(f"Top {top_n} LightGBM Feature Importances (Fold 1)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ensemble_top_feature_importances.png", bbox_inches="tight")
    plt.close()
    
    logger.info("=== MODEL TRAINING COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
