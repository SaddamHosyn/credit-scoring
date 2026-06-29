from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["figure.dpi"] = 110

try:
    from lightgbm import LGBMClassifier, log_evaluation
except ImportError as e:
    raise ImportError(
        "lightgbm is not installed. Install it with `pip install lightgbm`."
    ) from e

INPUT_FILE = Path("output/processed_data/application_train_engineered.csv")
TARGET_COL = "TARGET"
OUTPUT_DIR = Path("output/model_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Engineered dataset not found: {INPUT_FILE}")

print("=" * 70)
print("PHASE 1 - LIGHTGBM MODEL (ROC-AUC)")
print("=" * 70)


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


# 1) Load data
df = pd.read_csv(INPUT_FILE)

# Clean feature names for LightGBM compatibility
df.columns = clean_feature_names(df.columns)

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found after column cleaning.")

y = df[TARGET_COL].values
X = df.drop(columns=[TARGET_COL])

print(f"Data shape: {X.shape[0]:,} rows x {X.shape[1]:,} features")

# 2) Train/validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Train size: {X_train.shape[0]:,}  |  Valid size: {X_valid.shape[0]:,}")

# 3) Define LightGBM model
lgbm = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    random_state=42,
    n_jobs=-1,
)

# 4) Train
print("\nTraining LightGBM...")
callbacks = [log_evaluation(period=50)]

lgbm.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_metric="auc",
    callbacks=callbacks,
)

# 5) Evaluate
print("\nEvaluating LightGBM on validation set...")
y_valid_prob = lgbm.predict_proba(X_valid)[:, 1]
auc = roc_auc_score(y_valid, y_valid_prob)
print(f"Validation ROC-AUC (LightGBM): {auc:.4f}")

metrics_df = pd.DataFrame(
    {
        "model": ["lightgbm"],
        "metric": ["roc_auc"],
        "value": [auc],
    }
)
metrics_df.to_csv(OUTPUT_DIR / "lightgbm_metrics.csv", index=False)

# 6) ROC curve plot
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
plt.savefig(OUTPUT_DIR / "lightgbm_roc_curve.png", bbox_inches="tight")
plt.show()

# 7) Feature importance
print("\nSaving feature importances...")
importances = lgbm.feature_importances_

fi_df = pd.DataFrame(
    {
        "feature": X_train.columns,
        "importance": importances,
    }
).sort_values("importance", ascending=False)

fi_df.to_csv(OUTPUT_DIR / "lightgbm_feature_importances.csv", index=False)

top_n = 20
plt.figure(figsize=(10, 6))
sns.barplot(
    data=fi_df.head(top_n),
    x="importance",
    y="feature",
    palette="viridis",
)
plt.title(f"Top {top_n} LightGBM Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "lightgbm_top_feature_importances.png", bbox_inches="tight")
plt.show()

print("\nLightGBM artifacts saved to:", OUTPUT_DIR.resolve())
