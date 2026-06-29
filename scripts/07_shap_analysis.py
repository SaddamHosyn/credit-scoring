from pathlib import Path
import re

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

# ------------------- CONFIG -------------------
TRAIN_FILE = Path("output/processed_data/application_train_engineered.csv")
TARGET_COL = "TARGET"
RANDOM_STATE = 42
N_CLIENTS = 3  # number of example clients to export
# ------------------------------------------------


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


print("Loading engineered train data...")
df = pd.read_csv(TRAIN_FILE)

df.columns = clean_feature_names(df.columns)
y = df[TARGET_COL].values
X = df.drop(columns=[TARGET_COL])

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print("Training LightGBM for SHAP analysis...")
model = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
model.fit(X_train, y_train)

print("Computing SHAP values on validation set...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_valid)

# Handle binary-class output format changes across SHAP versions
# Either shap_values is:
# - a list [class0, class1]  (older behavior), or
# - a 2D array (n_samples, n_features) (newer behavior)
if isinstance(shap_values, list):
    shap_pos = shap_values[1]  # positive class
else:
    shap_pos = shap_values  # already 2D

# ----------------- GLOBAL SUMMARY PLOT -----------------
output_dir = Path("output/shap")
output_dir.mkdir(parents=True, exist_ok=True)

print("Saving global SHAP summary plot...")
shap.summary_plot(shap_pos, X_valid, max_display=20, show=False)
import matplotlib.pyplot as plt

plt.tight_layout()
plt.savefig(output_dir / "shap_summary_top20.png", bbox_inches="tight")
plt.close()

# ----------------- LOCAL EXPLANATIONS WITH LIGHTGBM SHAP -----------------
# Use LightGBM's own pred_contrib=True interface (returns SHAP values + bias term)
print("Selecting example clients...")

# Model predictions on validation set
y_valid_pred_prob = model.predict_proba(X_valid)[:, 1]
y_valid_pred = (y_valid_pred_prob >= 0.5).astype(int)

# Correct default (true 1, pred 1)
mask_correct_default = (y_valid == 1) & (y_valid_pred == 1)
# Incorrect prediction (true != pred)
mask_incorrect = y_valid != y_valid_pred
# Random non-default
mask_non_default = y_valid == 0


def first_index(mask):
    idx = np.where(mask)[0]
    return idx[0] if len(idx) > 0 else None


idx_correct = first_index(mask_correct_default)
idx_incorrect = first_index(mask_incorrect)
idx_non_default = first_index(mask_non_default)

examples = []
if idx_correct is not None:
    examples.append(("correct_default", idx_correct))
if idx_incorrect is not None:
    examples.append(("incorrect", idx_incorrect))
if idx_non_default is not None:
    examples.append(("non_default", idx_non_default))

print("Computing LightGBM SHAP values (pred_contrib=True) for local explanations...")
lgbm_shap = model.predict(X_valid, pred_contrib=True)
# Shape: (n_samples, n_features + 1), last column is bias term
feature_names = list(X_valid.columns) + ["bias"]

for label, local_idx in examples[:N_CLIENTS]:
    row = X_valid.iloc[[local_idx]]
    shap_row = lgbm_shap[local_idx]  # 1D array length = n_features+1

    # Split into feature contributions and bias
    contrib_values = shap_row[:-1]
    bias_value = shap_row[-1]

    # Save sorted top 10 contributions (by absolute impact)
    contrib_series = pd.Series(contrib_values, index=X_valid.columns)
    top10 = contrib_series.sort_values(key=lambda s: s.abs(), ascending=False).head(10)
    top10.to_csv(output_dir / f"shap_{label}_top10.csv")

    # For the report, we mainly need numbers, not plots;
    # but you can still build a simple bar plot if you want.
    # Here we just log:
    print(f"Saved top-10 SHAP contributions for {label} (index {local_idx}).")

print("Done. SHAP summary + local CSVs saved in:", output_dir.resolve())
