# Credit Scoring – Home Credit Default Risk

A transparent credit scoring project built on the Home Credit Default Risk dataset. The goal is to estimate the probability that a client will default on a loan, compare baseline and boosted models, and explain predictions at both the global and client level.

## Project objective

This project addresses a binary classification problem in consumer credit risk: predicting whether a customer will default on a loan. Because credit decisions affect both business performance and customers directly, the project focuses not only on predictive performance but also on interpretability.

The work follows two main principles:

- Build a model with strong discriminatory power, evaluated primarily with ROC–AUC.
- Explain the model globally and locally so the prediction logic is understandable.

## Dataset

Source: [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)

Main files used in this phase:

- `home-credit-default-risk/application_train.csv`
- `home-credit-default-risk/application_test.csv`
- `home-credit-default-risk/sample_submission.csv`

Additional relational tables are available in the dataset and can be integrated in later extensions, but the current phase focuses on the main application table.

## Repository structure

```text
.
├── .github/
│   └── workflows/
│       └── mlops.yml
├── app/
│   └── main.py
├── scripts/
│   ├── 02_feature_engineering.py
│   ├── 02_feature_engineering_test.py
│   ├── 03_model_training.py
│   ├── 04_logreg_learning_curves.py
│   ├── 05_lightgbm_model.py
│   ├── 06_kaggle_submission.py
│   ├── 07_shap_analysis.py
│   ├── 08_mlflow_training.py
│   ├── 09_hf_upload.py
│   └── 10_drift_monitoring.py
├── src/
│   ├── eda.py
│   └── preprocessing.py
├── tests/
│   └── test_app.py
├── .gitignore
├── 01_eda.ipynb
├── Dockerfile
├── README.md
├── requirements.txt
├── home-credit-default-risk/
│   ├── application_train.csv
│   ├── application_test.csv
│   └── ...
└── output/
    ├── processed_data/
    │   ├── application_train_engineered.csv
    │   └── application_test_engineered.csv
    ├── model_outputs/
    │   ├── validation_predictions.csv
    │   └── ...
    └── ...
```

## Workflow

### 1. Exploratory Data Analysis

EDA is available both as a notebook and as a script version:

- `01_eda.ipynb`
- `src/eda.py`

This stage covers:

- dataset shape and target balance,
- missing value inspection,
- univariate distributions,
- basic bivariate analysis against the target,
- correlation exploration.

### 2. Feature engineering

`scripts/02_feature_engineering_test.py` is the unified data preparation pipeline. To maximize predictive power, it aggregates and processes features across all 6 relational tables in the dataset:

*   **Main Application**: Extracts debt-to-income ratios, annuity-to-credit terms, employment percentages, and averages/products of external sources (`EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`).
*   **Bureau History**: Computes loan counts, total credit sums, active debt ratios, and status delays from `bureau.csv` and `bureau_balance.csv`.
*   **Previous Applications**: Aggregates loan approval/rejection counts, applied vs. received amounts, and payment counts from `previous_application.csv`.
*   **Installment Payments**: Tracks late payment delays (DPD) and installment underpayments from `installments_payments.csv`.
*   **POS Cash & Credit Cards**: Computes cash utilization rates, credit card utilization averages, and contract status tracking.

The script runs memory-efficiently by downcasting numerical types to prevent OOM issues and outputs a serving-time lookup store at `output/processed_data/secondary_features_lookup.csv`.

### 3. Baseline model

`03_model_training.py` trains a Logistic Regression baseline on a stratified 80/20 train–validation split.

Because the engineered dataset still contains missing values, median imputation and feature scaling are used before fitting Logistic Regression. The baseline is evaluated with ROC–AUC.

### 4. Learning curves and overfitting analysis

`04_logreg_learning_curves.py` generates learning curves for the Logistic Regression baseline.

The curves show that validation ROC–AUC stabilizes around 0.75 while training ROC–AUC converges slightly above it, around 0.76. The small gap suggests limited overfitting, while the relatively modest plateau indicates that the linear model is capacity-constrained for this problem.

### 5. Advanced Ensemble & Cross-Validation

`scripts/05_lightgbm_model.py` implements a robust ensembled validation pipeline:
- **Stratified 5-Fold Cross-Validation** to ensure stable, non-overfitting evaluation.
- **Model Ensembling**: Trains a combination of **LightGBM**, **XGBoost**, and **CatBoost** on each fold (15 models total).
- Predictions are averaged across folds and models, providing a major lift in score. Model weights are serialized into `output/model_outputs/model.pkl`.

### 6. Interpretability

`07_shap_analysis.py` generates SHAP-based explanations.

Interpretability is addressed at two levels:

- **Global interpretability** through feature importance and SHAP summary analysis.
- **Local interpretability** through client-level explanations for selected cases, including a correctly predicted default and a misclassified client.

### 7. Kaggle submission

`scripts/06_kaggle_submission.py` loads the trained 15-model ensemble and aligns test features to produce the optimized out-of-fold average predictions saved at:

- `output/kaggle/submission.csv`

This file can be uploaded directly to Kaggle for external evaluation.

### 8. Phase II: Production MLOps Setup

To transition this portfolio project into a production-grade system, Phase II adds a modern MLOps stack for model tracking, registry, cloud artifact storage, API serving, automated testing, and statistical drift monitoring:

#### A. Model Logging & Registry (MLflow)
- `08_mlflow_training.py` trains the LightGBM classifier and logs parameters, metrics (ROC-AUC and Kolmogorov-Smirnov statistic), validation prediction targets, and evaluation curves.
- The model is registered in the **MLflow Model Registry** under `credit_scoring_lgbm` and programmatically tagged with its metadata (e.g. `dataset_version: v1.0`, `feature_pipeline_version: v1.0`, `model_type: lightgbm`) using `MlflowClient`.

#### B. Remote Model Storage (Hugging Face)
- Large binary files (model weights and dataset definitions) are excluded from the main repository using `.gitignore` to keep the application repo lightweight.
- `09_hf_upload.py` uploads model pickle weights, feature list definitions, and model cards to a remote **Hugging Face Model Hub** repository.

#### C. Live Inference Serving (FastAPI & Uvicorn)
- `app/main.py` is a FastAPI serving application that exposes two endpoints:
  - `GET /health` for checkups.
  - `POST /score` to predict credit default risk on live applicants.
- On startup, the application fetches the latest model files from Hugging Face (falling back to local files if offline) and loads the precomputed `secondary_features_lookup.csv`.
- When scoring, it automatically merges raw payload records with precomputed history based on `SK_ID_CURR` (acting as a local feature store), and averages probabilities across the 15 ensembled models. It logs predictions using non-blocking FastAPI `BackgroundTasks`.

#### D. Live Score Drift Monitoring (PSI / PD drift)
- `10_drift_monitoring.py` analyzes shifts in probability of default (PD) and calculates the **Population Stability Index (PSI)** to detect population drift.
- It compares the training baseline predictions (`validation_predictions.csv`) against logged live predictions (`production_predictions.csv`).
- It supports customized CLI paths and gracefully falls back to mock simulation runs in clean environments (like automated CI runs).

#### E. Automated Workflows (GitHub Actions)
- `.github/workflows/mlops.yml` runs automated regression tests and checks the drift calculation script logic on every commit or PR to the main branch.

## Results

### Model comparison

| Model               | Validation ROC–AUC |
| ------------------- | -----------------: |
| Logistic Regression |              ~0.75 |
| LightGBM Baseline   |             0.7682 |
| LightGBM (OOF)      |             0.7852 |
| XGBoost (OOF)       |             0.7842 |
| CatBoost (OOF)      |             0.7752 |
| Ensemble (OOF)      |             0.7844 |

Integrating relational history from secondary tables combined with stratified 5-fold cross-validation and ensembling improved the model accuracy significantly by roughly 0.035 ROC–AUC over the initial linear baseline.

### Learning curve interpretation

The Logistic Regression learning curves indicate that the model is not strongly overfitting. Instead, it appears to be moderately underpowered relative to the complexity of the credit-risk problem, which motivates the transition to a boosted tree model.

### Global feature importance

The most important LightGBM features are dominated by external credit information and affordability-related ratios.

Top drivers include:

- `CREDIT_TERM`
- `EXT_SOURCE_3`
- `DAYS_BIRTH`
- `EXT_SOURCE_1`
- `DAYS_ID_PUBLISH`
- `EXT_SOURCES_MEAN`
- `DAYS_REGISTRATION`
- `DAYS_LAST_PHONE_CHANGE`
- `DAYS_EMPLOYED_PERCENT`
- `ANNUITY_INCOME_PERCENT`

These variables indicate that repayment burden, credit quality, and customer stability are central to default prediction.

### Local interpretability

For a correctly predicted defaulting client, SHAP values show that weak external credit information and an unfavorable repayment profile drive the high-risk prediction. In contrast, for a misclassified client, the model balances several moderate risk and protective signals, which illustrates the difficulty of perfectly identifying all defaults.

### Kaggle evaluation

The final LightGBM submission achieved:

| Evaluation          | ROC–AUC |
| ------------------- | ------: |
| Public leaderboard  | 0.76212 |
| Private leaderboard | 0.76056 |

The closeness between internal validation performance and Kaggle leaderboard performance indicates that the final model generalizes reasonably well to unseen customers.

## How to run

Create and activate a virtual environment:
```bash
python -m venv .venv
```

Activate the environment (Windows PowerShell):
```powershell
.venv/Scripts/Activate.ps1
```
*(For Git Bash, use `source .venv/Scripts/activate`)*

Install packages:
```bash
pip install -r requirements.txt
```

### Phase I Execution
```bash
python src/eda.py
python scripts/02_feature_engineering.py
python scripts/02_feature_engineering_test.py
python scripts/03_model_training.py
python scripts/04_logreg_learning_curves.py
python scripts/05_lightgbm_model.py
python scripts/07_shap_analysis.py
python scripts/06_kaggle_submission.py
```

### Phase II Execution (MLOps)
1. **Train and register model with MLflow:**
   ```bash
   python scripts/08_mlflow_training.py
   ```
2. **View MLflow Tracking UI:**
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```
3. **Upload model artifacts to Hugging Face (Optional):**
   ```bash
   export HF_REPO_ID="username/repo-name"
   export HF_TOKEN="your-token"
   python scripts/09_hf_upload.py
   ```
4. **Serve the FastAPI model local endpoint:**
   ```bash
   uvicorn app.main:app --port 8000 --reload
   ```
5. **Run tests:**
   ```bash
   python -m pytest tests/test_app.py
   ```
6. **Analyze probability drift:**
   ```bash
   python scripts/10_drift_monitoring.py
   ```

## Requirements

Main dependencies:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- lightgbm
- shap


