import json
import os
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# Import MLflow and LightGBM
import mlflow
import mlflow.lightgbm
from lightgbm import LGBMClassifier, log_evaluation

# Import shared preprocessing
from src.preprocessing import clean_feature_names

# Configuration
INPUT_FILE = Path("output/processed_data/application_train_engineered.csv")
TARGET_COL = "TARGET"
OUTPUT_DIR = Path("output/model_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Engineered dataset not found: {INPUT_FILE}")

# Set up MLflow to use local SQLite backend
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("credit-scoring")

def compute_ks_statistic(y_true, y_prob):
    """
    Computes the Kolmogorov-Smirnov (KS) statistic.
    It measures the maximum separation between defaults (1) and non-defaults (0) distributions.
    """
    df_ks = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df_ks = df_ks.sort_values(by="y_prob", ascending=False)
    
    total_defaults = df_ks["y_true"].sum()
    total_non_defaults = len(df_ks) - total_defaults
    
    if total_defaults == 0 or total_non_defaults == 0:
        return 0.0
        
    df_ks["cum_defaults"] = df_ks["y_true"].cumsum() / total_defaults
    df_ks["cum_non_defaults"] = (1 - df_ks["y_true"]).cumsum() / total_non_defaults
    
    ks_stat = (df_ks["cum_defaults"] - df_ks["cum_non_defaults"]).abs().max()
    return ks_stat

def main():
    print("=" * 70)
    print("PHASE 2 - LIGHTGBM TRAINING WITH MLFLOW & REGISTRY")
    print("=" * 70)

    # 1) Load data
    print("Loading engineered dataset...")
    df = pd.read_csv(INPUT_FILE)
    
    # Clean feature names for LightGBM compatibility
    df.columns = clean_feature_names(df.columns)
    
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found after column cleaning.")
    
    y = df[TARGET_COL].values
    X = df.drop(columns=[TARGET_COL])
    
    feature_list = list(X.columns)
    print(f"Data shape: {X.shape[0]:,} rows x {X.shape[1]:,} features")
    
    # 2) Save feature list locally for reference
    feature_list_path = OUTPUT_DIR / "feature_list.json"
    with open(feature_list_path, "w") as f:
        json.dump(feature_list, f, indent=4)
    print(f"Saved feature list to {feature_list_path}")

    # 3) Train/validation split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    
    print(f"Train size: {X_train.shape[0]:,}  |  Valid size: {X_valid.shape[0]:,}")

    # 4) Define model parameters
    params = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary",
        "random_state": 42,
        "n_jobs": -1,
    }

    # 5) MLflow tracking
    with mlflow.start_run() as run:
        print(f"Started MLflow run: {run.info.run_id}")
        
        # Log training parameters
        mlflow.log_params(params)
        
        # Log metadata tags
        mlflow.set_tags({
            "dataset_version": "v1.0",
            "feature_pipeline_version": "v1.0",
            "model_type": "lightgbm"
        })

        # Train model
        lgbm = LGBMClassifier(**params)
        callbacks = [log_evaluation(period=50)]
        
        print("\nTraining LightGBM...")
        lgbm.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_metric="auc",
            callbacks=callbacks,
        )

        # Predict probabilities
        print("\nEvaluating LightGBM...")
        y_valid_prob = lgbm.predict_proba(X_valid)[:, 1]
        
        # Calculate Metrics
        auc = roc_auc_score(y_valid, y_valid_prob)
        ks = compute_ks_statistic(y_valid, y_valid_prob)
        
        print(f"Validation ROC-AUC: {auc:.4f}")
        print(f"Validation KS Statistic: {ks:.4f}")
        
        # Log Metrics to MLflow
        mlflow.log_metric("validation_auc", auc)
        mlflow.log_metric("validation_ks", ks)

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_valid, y_valid_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f"LightGBM (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - LightGBM")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        
        roc_plot_path = OUTPUT_DIR / "mlflow_roc_curve.png"
        plt.savefig(roc_plot_path, bbox_inches="tight")
        plt.close()
        
        # Log ROC Plot to MLflow
        mlflow.log_artifact(str(roc_plot_path))

        # Plot Feature Importance
        importances = lgbm.feature_importances_
        fi_df = pd.DataFrame({
            "feature": X_train.columns,
            "importance": importances,
        }).sort_values("importance", ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=fi_df.head(20),
            x="importance",
            y="feature",
            palette="viridis",
        )
        plt.title("Top 20 LightGBM Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        
        fi_plot_path = OUTPUT_DIR / "mlflow_feature_importances.png"
        plt.savefig(fi_plot_path, bbox_inches="tight")
        plt.close()
        
        # Log Feature Importance Plot and list file to MLflow
        mlflow.log_artifact(str(fi_plot_path))
        mlflow.log_artifact(str(feature_list_path))

        # Log Model & Register in MLflow Registry
        print("\nLogging model and registering in Model Registry...")
        model_info = mlflow.lightgbm.log_model(
            lgb_model=lgbm,
            artifact_path="model",
            registered_model_name="credit_scoring_lgbm",
            input_example=X_train.head(1),
        )
        
        # Log the feature schema and version tags to model registry metadata
        print(f"Model successfully registered! URI: {model_info.model_uri}")
        
        # Save a local pickle for Hugging Face upload step
        import pickle
        model_pkl_path = OUTPUT_DIR / "model.pkl"
        with open(model_pkl_path, "wb") as f:
            pickle.dump(lgbm, f)
        print(f"Saved local pickle representation to {model_pkl_path}")

if __name__ == "__main__":
    main()
