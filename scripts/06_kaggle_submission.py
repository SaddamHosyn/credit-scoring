from pathlib import Path
import re

import pandas as pd
from lightgbm import LGBMClassifier

TRAIN_FILE = Path("output/processed_data/application_train_engineered.csv")
TEST_FILE = Path("output/processed_data/application_test_engineered.csv")
RAW_TEST_FILE = Path("home-credit-default-risk/application_test.csv")
TARGET_COL = "TARGET"

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


if not TRAIN_FILE.exists():
    raise FileNotFoundError(f"Missing train engineered file: {TRAIN_FILE}")
if not TEST_FILE.exists():
    raise FileNotFoundError(f"Missing test engineered file: {TEST_FILE}")
if not RAW_TEST_FILE.exists():
    raise FileNotFoundError(f"Missing raw test file: {RAW_TEST_FILE}")

print("Loading engineered train and test data...")
train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)
raw_test = pd.read_csv(RAW_TEST_FILE)

print("Cleaning feature names for LightGBM compatibility...")
train.columns = clean_feature_names(train.columns)
test.columns = clean_feature_names(test.columns)

if TARGET_COL not in train.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in train file.")

y = train[TARGET_COL].values
X = train.drop(columns=[TARGET_COL])

test = test.reindex(columns=X.columns, fill_value=0)

print("Training final LightGBM model on full training data...")
model = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    random_state=42,
    n_jobs=-1,
)
model.fit(X, y)

print("Predicting probabilities for Kaggle test set...")
test_probs = model.predict_proba(test)[:, 1]

submission = pd.DataFrame(
    {
        "SK_ID_CURR": raw_test["SK_ID_CURR"],
        "TARGET": test_probs,
    }
)

submission.to_csv(SUB_FILE, index=False)

print("Submission file saved to:", SUB_FILE.resolve())
print(submission.head())
