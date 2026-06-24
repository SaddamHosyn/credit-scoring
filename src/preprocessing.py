import re
import numpy as np
import pandas as pd

def clean_feature_names(columns):
    """
    Cleans feature names for LightGBM compatibility.
    Removes special characters and handles duplicate column names.
    """
    cleaned = []
    seen = {}

    for col in columns:
        # Keep only alphanumeric and underscore characters
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

def preprocess_features(df: pd.DataFrame, expected_features: list = None) -> pd.DataFrame:
    """
    Applies the Phase 1 feature engineering steps to the input DataFrame.
    If expected_features is provided, reindexes the columns to align with it.
    """
    df = df.copy()

    # 1. Handle Known Anomalies
    if "DAYS_EMPLOYED" in df.columns:
        # replace anomalous value 365243 with NaN
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
        df["DAYS_EMPLOYED_ANOM"] = df["DAYS_EMPLOYED"].isna().astype(int)
    else:
        df["DAYS_EMPLOYED_ANOM"] = 0

    # 2. Domain Knowledge Feature Engineering
    if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["CREDIT_INCOME_PERCENT"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    else:
        df["CREDIT_INCOME_PERCENT"] = np.nan

    if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["ANNUITY_INCOME_PERCENT"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    else:
        df["ANNUITY_INCOME_PERCENT"] = np.nan

    if "AMT_ANNUITY" in df.columns and "AMT_CREDIT" in df.columns:
        df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    else:
        df["CREDIT_TERM"] = np.nan

    if "DAYS_EMPLOYED" in df.columns and "DAYS_BIRTH" in df.columns:
        df["DAYS_EMPLOYED_PERCENT"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    else:
        df["DAYS_EMPLOYED_PERCENT"] = np.nan

    # EXT_SOURCE features aggregations
    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
    if ext_cols:
        df["EXT_SOURCES_MEAN"] = df[ext_cols].mean(axis=1)
        df["EXT_SOURCES_MIN"] = df[ext_cols].min(axis=1)
        df["EXT_SOURCES_MAX"] = df[ext_cols].max(axis=1)
        df["EXT_SOURCES_PROD"] = df[ext_cols].prod(axis=1)
    else:
        df["EXT_SOURCES_MEAN"] = np.nan
        df["EXT_SOURCES_MIN"] = np.nan
        df["EXT_SOURCES_MAX"] = np.nan
        df["EXT_SOURCES_PROD"] = np.nan

    # 3. Categorical Encoding (One-Hot Encoding)
    # We apply pd.get_dummies to all categorical (object/category) columns
    df = pd.get_dummies(df, dummy_na=False)

    # 4. Clean column names to prevent LightGBM issues with JSON/Special characters
    df.columns = clean_feature_names(df.columns)

    # 5. Column Alignment
    if expected_features is not None:
        df = df.reindex(columns=expected_features, fill_value=0)

    return df
