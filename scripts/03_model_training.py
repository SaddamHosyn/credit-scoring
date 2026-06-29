from pathlib import Path
from sklearn.impute import SimpleImputer

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["figure.dpi"] = 110

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
INPUT_FILE = Path("output/processed_data/application_train_engineered.csv")
TARGET_COL = "TARGET"
OUTPUT_DIR = Path("output/model_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Engineered dataset not found: {INPUT_FILE}")

print("=" * 70)
print("PHASE 1 - BASELINE MODEL TRAINING (LOGISTIC REGRESSION, AUC)")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------------
print("\n1) Loading engineered data...")
df = pd.read_csv(INPUT_FILE)
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

y = df[TARGET_COL].values
X = df.drop(columns=[TARGET_COL])

print(f"Number of features: {X.shape[1]}")

# -----------------------------------------------------------------------------
# 2. TRAIN/VALIDATION SPLIT
# -----------------------------------------------------------------------------
print("\n2) Creating train/validation split (80/20, stratified)...")
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Train size: {X_train.shape[0]:,}  |  Valid size: {X_valid.shape[0]:,}")

# -----------------------------------------------------------------------------
# 3. IMPUTATION + SCALING FOR LOGISTIC REGRESSION
# -----------------------------------------------------------------------------
print("\n3) Imputing missing values and scaling features for Logistic Regression...")

# Simple strategy: replace NaN with median of each column
imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)
X_valid_imputed = imputer.transform(X_valid)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_valid_scaled = scaler.transform(X_valid_imputed)

# -----------------------------------------------------------------------------
# 4. TRAIN BASELINE MODEL
# -----------------------------------------------------------------------------
print("\n4) Training Logistic Regression baseline model...")
log_reg = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    n_jobs=-1,
    class_weight="balanced",  # handle class imbalance
)
log_reg.fit(X_train_scaled, y_train)

# -----------------------------------------------------------------------------
# 5. EVALUATE WITH ROC-AUC
# -----------------------------------------------------------------------------
print("\n5) Evaluating on validation set (ROC-AUC)...")
y_valid_prob = log_reg.predict_proba(X_valid_scaled)[:, 1]
y_valid_pred = (y_valid_prob >= 0.5).astype(int)

auc = roc_auc_score(y_valid, y_valid_prob)
print(f"Validation ROC-AUC: {auc:.4f}")

# Save basic evaluation results
metrics_df = pd.DataFrame(
    {
        "metric": ["roc_auc"],
        "value": [auc],
    }
)
metrics_df.to_csv(OUTPUT_DIR / "baseline_logreg_metrics.csv", index=False)

# -----------------------------------------------------------------------------
# 6. PLOT ROC CURVE
# -----------------------------------------------------------------------------
print("6) Plotting ROC curve...")
fpr, tpr, thresholds = roc_curve(y_valid, y_valid_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"LogReg (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression Baseline")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "baseline_logreg_roc_curve.png", bbox_inches="tight")
plt.show()

# -----------------------------------------------------------------------------
# 7. SAVE VALIDATION PREDICTIONS FOR INSPECTION
# -----------------------------------------------------------------------------
print("7) Saving validation predictions...")
valid_out = X_valid.copy()
valid_out["TARGET"] = y_valid
valid_out["PRED_PROB"] = y_valid_prob
valid_out["PRED_LABEL"] = y_valid_pred

valid_out.to_csv(OUTPUT_DIR / "baseline_logreg_valid_predictions.csv", index=False)

print("\nDone. Baseline model artifacts saved to:", OUTPUT_DIR.resolve())
