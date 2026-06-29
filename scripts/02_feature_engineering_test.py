import os
import numpy as np
import pandas as pd

TRAIN_INPUT = "home-credit-default-risk/application_train.csv"
TEST_INPUT = "home-credit-default-risk/application_test.csv"
TRAIN_ENGINEERED = "output/processed_data/application_train_engineered.csv"
TEST_ENGINEERED = "output/processed_data/application_test_engineered.csv"
TARGET_COL = "TARGET"

os.makedirs("output/processed_data", exist_ok=True)

print("Loading train and test datasets...")
train = pd.read_csv(TRAIN_INPUT)
test = pd.read_csv(TEST_INPUT)

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

train_target = train[TARGET_COL].copy()
train_features = train.drop(columns=[TARGET_COL]).copy()

combined = pd.concat([train_features, test], axis=0, ignore_index=True)

print("Applying feature engineering to combined train/test data...")

combined["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)
combined["DAYS_EMPLOYED_ANOM"] = combined["DAYS_EMPLOYED"].isna().astype(int)

combined["CREDIT_INCOME_PERCENT"] = (
    combined["AMT_CREDIT"] / combined["AMT_INCOME_TOTAL"]
)
combined["ANNUITY_INCOME_PERCENT"] = (
    combined["AMT_ANNUITY"] / combined["AMT_INCOME_TOTAL"]
)
combined["CREDIT_TERM"] = combined["AMT_ANNUITY"] / combined["AMT_CREDIT"]
combined["DAYS_EMPLOYED_PERCENT"] = combined["DAYS_EMPLOYED"] / combined["DAYS_BIRTH"]

ext_cols = [
    c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in combined.columns
]
if ext_cols:
    combined["EXT_SOURCES_MEAN"] = combined[ext_cols].mean(axis=1)
    combined["EXT_SOURCES_MIN"] = combined[ext_cols].min(axis=1)
    combined["EXT_SOURCES_MAX"] = combined[ext_cols].max(axis=1)
    combined["EXT_SOURCES_PROD"] = combined[ext_cols].prod(axis=1)

print("One-hot encoding categorical variables...")
combined = pd.get_dummies(combined, dummy_na=False)

train_rows = len(train_features)
train_eng = combined.iloc[:train_rows, :].copy()
test_eng = combined.iloc[train_rows:, :].copy()

train_eng[TARGET_COL] = train_target.values

print("Aligning columns with existing engineered train file...")
if os.path.exists(TRAIN_ENGINEERED):
    existing_train = pd.read_csv(TRAIN_ENGINEERED)
    feature_cols = [c for c in existing_train.columns if c != TARGET_COL]

    train_eng = train_eng.reindex(columns=feature_cols, fill_value=0)
    train_eng[TARGET_COL] = train_target.values
    test_eng = test_eng.reindex(columns=feature_cols, fill_value=0)

train_eng.to_csv(TRAIN_ENGINEERED, index=False)
test_eng.to_csv(TEST_ENGINEERED, index=False)

print("Saved files:")
print(f"  {TRAIN_ENGINEERED}")
print(f"  {TEST_ENGINEERED}")
print(f"Final train engineered shape: {train_eng.shape}")
print(f"Final test engineered shape: {test_eng.shape}")
