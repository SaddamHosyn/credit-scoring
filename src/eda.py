from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["figure.dpi"] = 110

FILE_PATH = Path("home-credit-default-risk/application_train.csv")
TARGET_COL = "TARGET"
OUTPUT_DIR = Path("output/eda_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not FILE_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {FILE_PATH}")


def save_fig(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    plt.show()
    plt.close()


print("=" * 80)
print("HOME CREDIT DEFAULT RISK - EDA")
print("=" * 80)

df = pd.read_csv(FILE_PATH)

print("\n1) DATASET OVERVIEW")
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
print("\nFirst 10 columns:")
print(df.columns[:10].tolist())

overview_df = pd.DataFrame(
    {
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "missing_count": df.isna().sum().values,
        "missing_pct": (df.isna().mean().values * 100).round(2),
        "n_unique": df.nunique(dropna=True).values,
    }
)
overview_df.to_csv(OUTPUT_DIR / "dataset_overview.csv", index=False)

print("\n2) TARGET DISTRIBUTION")
target_counts = df[TARGET_COL].value_counts().sort_index()
target_pct = df[TARGET_COL].value_counts(normalize=True).sort_index().mul(100).round(2)
target_summary = pd.DataFrame(
    {
        "TARGET": target_counts.index,
        "label": ["non-default" if x == 0 else "default" for x in target_counts.index],
        "count": target_counts.values,
        "percentage": target_pct.values,
    }
)
print(target_summary.to_string(index=False))
target_summary.to_csv(OUTPUT_DIR / "target_distribution.csv", index=False)

plt.figure()
ax = sns.countplot(data=df, x=TARGET_COL, palette="Set2")
plt.title("Target Distribution")
plt.xlabel("TARGET (0 = non-default, 1 = default)")
plt.ylabel("Count")
for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height()):,}",
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha="center",
        va="bottom",
        fontsize=10,
        xytext=(0, 4),
        textcoords="offset points",
    )
save_fig("target_distribution.png")

print("\n3) DATA QUALITY CHECKS")
missing_df = (
    df.isnull()
    .sum()
    .reset_index()
    .rename(columns={"index": "feature", 0: "missing_count"})
)
missing_df["missing_pct"] = (missing_df["missing_count"] / len(df) * 100).round(2)
missing_df = missing_df[missing_df["missing_count"] > 0].sort_values(
    by="missing_pct", ascending=False
)
print(f"Columns with missing values: {missing_df.shape[0]}")
print(f"Duplicate rows: {df.duplicated().sum():,}")
missing_df.to_csv(OUTPUT_DIR / "missing_values_summary.csv", index=False)

if not missing_df.empty:
    top_missing = missing_df.head(20)
    plt.figure(figsize=(12, 7))
    sns.barplot(data=top_missing, x="missing_pct", y="feature", palette="viridis")
    plt.title("Top 20 Columns by Missing Percentage")
    plt.xlabel("Missing Percentage")
    plt.ylabel("Feature")
    save_fig("top_missing_values.png")

print("\n4) FEATURE TYPES")
num_cols = df.select_dtypes(include=[np.number]).columns.drop(
    TARGET_COL, errors="ignore"
)
cat_cols = df.select_dtypes(include=["object"]).columns
print(f"Numerical columns: {len(num_cols)}")
print(f"Categorical columns: {len(cat_cols)}")

num_summary = df[num_cols].describe().T
num_summary.to_csv(OUTPUT_DIR / "numerical_summary.csv")

cat_summary = pd.DataFrame(
    {
        "feature": cat_cols,
        "n_unique": [df[c].nunique(dropna=True) for c in cat_cols],
        "top_value": [
            (
                df[c].mode(dropna=True).iloc[0]
                if not df[c].mode(dropna=True).empty
                else np.nan
            )
            for c in cat_cols
        ],
        "top_freq": [
            (
                df[c].value_counts(dropna=True).iloc[0]
                if not df[c].value_counts(dropna=True).empty
                else np.nan
            )
            for c in cat_cols
        ],
    }
)
cat_summary.to_csv(OUTPUT_DIR / "categorical_summary.csv", index=False)

print("\n5) UNIVARIATE ANALYSIS")
selected_num_cols = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
]
selected_num_cols = [c for c in selected_num_cols if c in df.columns]

for col in selected_num_cols:
    plt.figure()
    sns.histplot(df[col].dropna(), bins=40, kde=True, color="#2a9d8f")
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    save_fig(f"hist_{col.lower()}.png")

print("\n6) BIVARIATE ANALYSIS VS TARGET")
for col in selected_num_cols:
    plt.figure()
    sns.boxplot(data=df, x=TARGET_COL, y=col, palette="Set3")
    plt.title(f"{col} by TARGET")
    plt.xlabel("TARGET")
    plt.ylabel(col)
    save_fig(f"boxplot_{col.lower()}_by_target.png")

selected_cat_cols = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
]
selected_cat_cols = [c for c in selected_cat_cols if c in df.columns]

default_rate_tables = []
for col in selected_cat_cols:
    rate_df = (
        df.groupby(col, dropna=False)[TARGET_COL]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"mean": "default_rate"})
        .sort_values(by="default_rate", ascending=False)
    )
    rate_df["default_rate"] = (rate_df["default_rate"] * 100).round(2)
    rate_df.to_csv(OUTPUT_DIR / f"default_rate_by_{col.lower()}.csv", index=False)
    default_rate_tables.append(rate_df.assign(feature=col))

    plt.figure(figsize=(12, 6))
    sns.barplot(data=rate_df.head(15), x="default_rate", y=col, palette="magma")
    plt.title(f"Default Rate by {col}")
    plt.xlabel("Default Rate (%)")
    plt.ylabel(col)
    save_fig(f"default_rate_by_{col.lower()}.png")

if default_rate_tables:
    pd.concat(default_rate_tables, ignore_index=True).to_csv(
        OUTPUT_DIR / "default_rate_summary_selected_categoricals.csv", index=False
    )

print("\n7) CORRELATION ANALYSIS")
corr_cols = [
    c
    for c in [
        "TARGET",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
    ]
    if c in df.columns
]
corr = df[corr_cols].corr(numeric_only=True)
corr.to_csv(OUTPUT_DIR / "correlation_matrix_selected_features.csv")

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=False)
plt.title("Correlation Heatmap - Selected Features")
save_fig("correlation_heatmap_selected_features.png")

# 8) BASIC FINDINGS
print("\n8) BASIC FINDINGS")
findings = []
findings.append(f"Dataset contains {df.shape[0]:,} rows and {df.shape[1]:,} columns.")
findings.append(
    f"Target is imbalanced: {target_pct.loc[0]:.2f}% non-default vs {target_pct.loc[1]:.2f}% default."
)
findings.append(f"There are {missing_df.shape[0]} columns with missing values.")
findings.append(f"Duplicate rows found: {df.duplicated().sum():,}.")
if "EXT_SOURCE_2" in df.columns:
    findings.append(
        "External source features such as EXT_SOURCE variables should be inspected closely because they are often highly predictive."
    )
if "DAYS_EMPLOYED" in df.columns:
    findings.append(
        "DAYS_EMPLOYED requires special checking because this dataset is known to contain unusual employment-day values."
    )

with open(OUTPUT_DIR / "eda_findings.txt", "w", encoding="utf-8") as f:
    for i, item in enumerate(findings, 1):
        line = f"{i}. {item}"
        print(line)
        f.write(line + "\n")

# 9) MISSING VALUE DEEP DIVE (from notebook Cell 8)
print("\n9) MISSING VALUE DEEP DIVE")
print("--- Understanding Missing Data ---")

if "FLAG_OWN_CAR" in df.columns and "OWN_CAR_AGE" in df.columns:
    car_no_car = df[df["FLAG_OWN_CAR"] == "N"]["OWN_CAR_AGE"].isna().all()
    print(
        f"Is OWN_CAR_AGE always missing when the client does not own a car? {car_no_car}"
    )
else:
    print(
        "Columns FLAG_OWN_CAR or OWN_CAR_AGE not found; skipping structural car-age check."
    )

ext_cols = [
    col for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if col in df.columns
]
if ext_cols:
    ext_missing_pct = (df[ext_cols].isnull().sum() / len(df) * 100).round(2)
    ext_missing_pct.to_csv(
        OUTPUT_DIR / "ext_source_missing_pct.csv", header=["missing_pct"]
    )
    print("\nMissing value percentage in EXT_SOURCE features:")
    print((ext_missing_pct.astype(str) + "%").to_string())
else:
    print("No EXT_SOURCE columns found; skipping EXT_SOURCE missingness analysis.")

# 10) CATEGORICAL FEATURES VS DEFAULT RATE (from notebook Cell 9)
print("\n10) CATEGORICAL FEATURES VS DEFAULT RATE")
print("\n--- Default Rate by Category (compact version) ---")


def plot_default_rate(col: str) -> None:
    if col not in df.columns:
        print(f"Column {col} not found; skipping.")
        return

    summary = (
        df.groupby(col, dropna=False)[TARGET_COL]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"mean": "default_rate"})
    )
    summary["default_rate"] = summary["default_rate"] * 100

    summary = summary[summary["count"] > 100].sort_values(
        "default_rate", ascending=False
    )

    summary.to_csv(OUTPUT_DIR / f"default_rate_compact_{col.lower()}.csv", index=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=summary, x="default_rate", y=col, palette="Reds_r")
    plt.title(f"Default Rate (%) by {col}")
    plt.xlabel("Default Rate (%)")
    plt.ylabel(col)
    save_fig(f"default_rate_compact_{col.lower()}.png")


categorical_cols_compact = [
    "CODE_GENDER",
    "NAME_EDUCATION_TYPE",
    "NAME_INCOME_TYPE",
    "OCCUPATION_TYPE",
]
for c in categorical_cols_compact:
    plot_default_rate(c)

print("\nEDA files saved to:", OUTPUT_DIR.resolve())
