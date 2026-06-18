from pathlib import Path
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["figure.dpi"] = 110

INPUT_FILE = Path("output/processed_data/application_train_engineered.csv")
TARGET_COL = "TARGET"
OUTPUT_DIR = Path("output/model_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Engineered dataset not found: {INPUT_FILE}")

print("=" * 70)
print("PHASE 1 - LOGISTIC REGRESSION LEARNING CURVES (ROC–AUC)")
print("=" * 70)

# 1) Load data
df = pd.read_csv(INPUT_FILE)
y = df[TARGET_COL].values
X = df.drop(columns=[TARGET_COL])

print(f"Data shape: {X.shape[0]:,} rows x {X.shape[1]:,} features")

# 2) Imputation + standardization
print("\nImputing missing values and scaling features for learning curves...")

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 3) Define model and CV
log_reg = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    class_weight="balanced",  # drop n_jobs; it has no effect and gives warnings
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4) Learning curve (using ROC–AUC scoring)
print("\nComputing learning curves (this may take a bit)...")
train_sizes, train_scores, valid_scores = learning_curve(
    estimator=log_reg,
    X=X_scaled,
    y=y,
    train_sizes=np.linspace(0.1, 1.0, 6),
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    shuffle=True,
    random_state=42,
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
valid_mean = valid_scores.mean(axis=1)
valid_std = valid_scores.std(axis=1)

# Save numeric results
lc_df = pd.DataFrame(
    {
        "train_size": train_sizes,
        "train_mean_auc": train_mean,
        "train_std_auc": train_std,
        "valid_mean_auc": valid_mean,
        "valid_std_auc": valid_std,
    }
)
lc_df.to_csv(OUTPUT_DIR / "logreg_learning_curve.csv", index=False)

# 5) Plot
plt.figure()
plt.plot(train_sizes, train_mean, "o-", label="Training AUC")
plt.fill_between(
    train_sizes,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.15,
)
plt.plot(train_sizes, valid_mean, "o-", label="Validation AUC")
plt.fill_between(
    train_sizes,
    valid_mean - valid_std,
    valid_mean + valid_std,
    alpha=0.15,
)
plt.xlabel("Training set size")
plt.ylabel("ROC–AUC")
plt.title("Learning Curves - Logistic Regression")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "logreg_learning_curves.png", bbox_inches="tight")
plt.show()

print(
    "\nLearning curves saved to:", (OUTPUT_DIR / "logreg_learning_curves.png").resolve()
)
print("Numeric results saved to:", (OUTPUT_DIR / "logreg_learning_curve.csv").resolve())
