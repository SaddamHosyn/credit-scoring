import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# Configuration
INPUT_FILE = 'home-credit-default-risk/application_train.csv'
OUTPUT_DIR = 'output/processed_data'
OUTPUT_FILE = f'{OUTPUT_DIR}/application_train_engineered.csv'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PHASE 1 - FEATURE ENGINEERING PIPELINE")
print("=" * 60)

# 1. Load Data
print("1. Loading raw data...")
df = pd.read_csv(INPUT_FILE)
print(f"Initial shape: {df.shape}")

# 2. Handle Known Anomalies
print("\n2. Handling dataset anomalies...")
# The value 365243 in DAYS_EMPLOYED means "Pensioner" (unemployed). It ruins models.
anomaly_count = len(df[df['DAYS_EMPLOYED'] == 365243])
print(f"Found {anomaly_count} anomalous DAYS_EMPLOYED values. Replacing with NaN.")
df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

# Create a boolean flag for these anomalies (often a good predictor)
df['DAYS_EMPLOYED_ANOM'] = df['DAYS_EMPLOYED'].isna().astype(int)

# 3. Domain Knowledge Feature Engineering
print("\n3. Creating domain-specific financial features...")
# Ratios are often more predictive than absolute amounts
df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT'] # How long the loan will take
df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH'] # % of life spent working

# EXT_SOURCE features are the strongest predictors. Create aggregations.
ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
# Only keep columns that actually exist in the dataframe
ext_cols = [c for c in ext_cols if c in df.columns]

if ext_cols:
    print(f"Creating aggregations for external sources: {ext_cols}")
    df['EXT_SOURCES_MEAN'] = df[ext_cols].mean(axis=1)
    df['EXT_SOURCES_MIN'] = df[ext_cols].min(axis=1)
    df['EXT_SOURCES_MAX'] = df[ext_cols].max(axis=1)
    df['EXT_SOURCES_PROD'] = df[ext_cols].prod(axis=1)

# 4. Handle Categorical Variables
print("\n4. Encoding categorical variables...")
categorical_cols = df.select_dtypes(include=['object']).columns

# Label Encoding for binary categories (2 unique values)
le = LabelEncoder()
le_count = 0

for col in categorical_cols:
    if df[col].nunique() <= 2:
        df[col] = le.fit_transform(df[col].astype(str))
        le_count += 1

print(f"Label Encoded {le_count} binary categorical columns.")

# One-Hot Encoding for remaining categorical columns (>2 unique values)
print("Applying One-Hot Encoding to remaining categorical columns...")
df = pd.get_dummies(df, drop_first=True)

# 5. Final Checks and Save
print("\n5. Saving engineered dataset...")
print(f"Final engineered shape: {df.shape}")

df.to_csv(OUTPUT_FILE, index=False)
print(f"Success! Engineered dataset saved to: {OUTPUT_FILE}")
