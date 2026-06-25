# Credit-Risk-Modelling-Loan-Default-Prediction

> End-to-end binary credit risk pipeline: data ingestion → cleaning → EDA → machine learning → risk scoring → business analytics.

Works with **synthetic data out of the box** (zero setup) and with the **real Lending Club dataset** (2007–2018, ~2.26 million loans) via a single flag.

---

## Table of contents

1. [Overview](#overview)
2. [Quick start](#quick-start)
3. [Getting the real dataset](#getting-the-real-dataset)
4. [Pipeline stages](#pipeline-stages)
5. [Features used](#features-used)
6. [Model performance](#model-performance)
7. [Risk tiers](#risk-tiers)
8. [Business metrics](#business-metrics)
9. [CLI options](#cli-options)
10. [Output files](#output-files)
11. [Project structure](#project-structure)
12. [Running tests](#running-tests)
13. [Methodology](#methodology)
14. [Requirements](#requirements)
15. [License](#license)

---

## Overview

This project builds a **binary default classifier** on Lending Club loan data, converts predicted default probabilities into actionable risk tiers, and calculates portfolio-level business metrics (Expected Loss, revenue, and an optimal approval threshold).

**Pipeline at a glance:**

```
Raw CSV / synthetic data
        ↓
  Data cleaning & feature engineering  (impute, encode, engineer 16 features)
        ↓
  Exploratory Data Analysis             (4 figures)
        ↓
  Logistic Regression + Random Forest   (class-balanced training, ROC-AUC eval)
        ↓
  Risk scoring                          (Low / Medium / High tiers)
        ↓
  Business layer                        (EL = PD × LGD × EAD, threshold sweep)
        ↓
  Outputs: figures + scored CSV + metrics JSON
```

---

## Quick start

```bash
# 1. Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn

# 2a. Run with synthetic data (no download needed)
python credit_risk_pipeline_final.py

# 2b. Run with real Lending Club CSV
python credit_risk_pipeline_final.py --data data/accepted_2007_to_2018Q4.csv

# 2c. Fast test on 100K rows of real data
python credit_risk_pipeline_final.py --data data/loan.csv --nrows 100000

# 2d. Custom output folder
python credit_risk_pipeline_final.py --output results/
```

All outputs are written to `outputs/` (or the folder you specify).

---

## Getting the real dataset

### Option A — Kaggle CLI (recommended)

```bash
pip install kaggle

# 1. Get your API key from kaggle.com → Account → Create New API Token
#    Save kaggle.json to ~/.kaggle/kaggle.json and chmod 600 it

# 2. Full dataset: 2007–2018, ~2.26M rows, ~1.6 GB uncompressed
kaggle datasets download -d wordsforthewise/lending-club -p data/ --unzip
# → data/accepted_2007_to_2018Q4.csv

# 3. Smaller alternative: 2007–2015, ~890K rows, ~430 MB
kaggle datasets download -d adarshsng/lending-club-loan-data-csv -p data/ --unzip
# → data/loan.csv
```

### Option B — Zenodo (pre-cleaned, 1.35M rows)

A pre-filtered version (Fully Paid / Charged Off only, post-application columns) is available at:
`https://zenodo.org/records/11295916`

### Real data quirks — handled automatically

| Quirk | How it is handled |
|-------|------------------|
| `int_rate` stored as `"10.65%"` | Strip `%`, cast to `float` |
| `revol_util` stored as `"83.7%"` | Same |
| Junk 2nd header row in some exports | Auto-detected and skipped |
| `emp_length` values `"n/a"` / `NaN` | Normalised to `"Unknown"` |
| `home_ownership` values `"ANY"` / `"NONE"` | Merged into `"OTHER"` |
| Transitory statuses (`Current`, `In Grace Period`) | Dropped — only `Fully Paid` and `Charged Off` are kept |
| Missing `fico_range_low` column | Estimated from `grade` |
| Mixed-type numeric columns | `pd.to_numeric(..., errors="coerce")` |

---

## Pipeline stages

| # | Stage | What it does |
|---|-------|-------------|
| 1 | **Data collection** | Generates a realistic synthetic dataset (15K loans) OR loads and cleans a real Lending Club CSV |
| 2 | **Preprocessing** | Imputes missing values, engineers 5 new features, label-encodes categoricals, median-imputes residuals |
| 3 | **EDA** | Produces income/interest rate/loan amount distributions and default rate by employment length and grade |
| 4 | **Modeling** | Trains Logistic Regression and Random Forest on an upsampled (50/50) training set; evaluates with ROC-AUC and Average Precision |
| 5 | **Risk scoring** | Assigns each loan a predicted default probability and a Low / Medium / High tier |
| 6 | **Business layer** | Computes Expected Loss, revenue estimate, and sweeps 50 approval thresholds to find the profit-maximising cutoff |

---

## Features used

| Feature | Source | Description |
|---------|--------|-------------|
| `loan_amnt` | Raw | Requested loan amount |
| `int_rate` | Raw | Interest rate (%) |
| `term_num` | Engineered | Loan term in months (36 or 60) |
| `grade_num` | Engineered | Loan grade A–G encoded 1–7 |
| `emp_length_num` | Engineered | Employment length in years (numeric) |
| `home_enc` | Encoded | Home ownership status |
| `annual_inc` | Raw | Annual income |
| `purpose_enc` | Encoded | Loan purpose |
| `dti` | Raw | Debt-to-income ratio |
| `delinq_2yrs` | Raw | Delinquencies in past 2 years |
| `fico_range_low` | Raw | Lower FICO score |
| `open_acc` | Raw | Number of open credit lines |
| `pub_rec` | Raw | Derogatory public records |
| `revol_util` | Raw | Revolving credit utilisation (%) |
| `inc_to_loan` | Engineered | `annual_inc / loan_amnt` |
| `monthly_payment_est` | Engineered | `loan_amnt × int_rate / 12` |

---

## Model performance

| Model | AUC — Synthetic | AUC — Real data (2007–2015) |
|-------|----------------|-----------------------------|
| Logistic Regression | 0.66 | ~0.68–0.70 |
| Random Forest | 0.43 | ~0.70–0.74 |

> The Random Forest underperforms on the synthetic dataset due to the low default rate (~1.6%) and the simple linear structure of the synthetic data generating process. On real data, RF typically outperforms LR once sufficient defaulted examples exist.

**Class balancing:** the training set is upsampled to a 50/50 ratio using `sklearn.utils.resample`. The test set is always kept at the original distribution for unbiased evaluation.

---

## Risk tiers

Predicted default probabilities are bucketed into three tiers:

| Tier | Probability of Default | Typical share |
|------|----------------------|---------------|
| **Low** | < 15% | ~10–30% |
| **Medium** | 15%–35% | ~55–65% |
| **High** | ≥ 35% | ~10–15% |

Thresholds are configurable via `RISK_THRESHOLDS` in the script.

---

## Business metrics

### Expected Loss

```
EL = PD × LGD × EAD
```

| Symbol | Meaning | Default value |
|--------|---------|--------------|
| PD | Probability of Default | model output |
| LGD | Loss Given Default | 1 − 0.40 = **0.60** (40% recovery assumed) |
| EAD | Exposure at Default | `loan_amnt` |

### Revenue estimate

```
Revenue ≈ loan_amnt × (int_rate / 100) × (term_months / 12) × (1 − PD)
```

### Approval threshold optimisation

A sweep across 50 thresholds (5%–60%) evaluates `net profit = Revenue − EL` at each cutoff. The optimal threshold maximises portfolio-level net profit.

| Metric | Value (synthetic run) |
|--------|-----------------------|
| Optimal threshold | 0.23 |
| Approval rate | 100% |
| Estimated net profit | $9,443,341 |
| Total EL — Low tier | $519,071 |
| Total EL — Medium tier | $3,877,569 |

---

## CLI options

```
python credit_risk_pipeline_final.py [OPTIONS]

  --data    PATH    Path to real Lending Club CSV
                    (omit to use synthetic data)
  --nrows   INT     Limit rows read from CSV
                    (useful for large files; e.g. --nrows 100000)
  --output  PATH    Output directory
                    (default: outputs/)
  --n       INT     Synthetic dataset size
                    (default: 15000; ignored when --data is set)
  --seed    INT     Random seed
                    (default: 42)
```

---

## Output files

| File | Type | Description |
|------|------|-------------|
| `scored_loans.csv` | CSV | Test-set loans with PD score, risk tier, EL, revenue estimate |
| `threshold_sweep.csv` | CSV | Net profit, approval rate, and defaults at each of 50 thresholds |
| `model_metrics.json` | JSON | AUC scores, optimal threshold, tier counts, data source |
| `eda_overview.png` | Figure | 2×2 grid: income, interest rate, loan amount, employment length |
| `correlation_heatmap.png` | Figure | Lower-triangle feature correlation matrix |
| `default_rate_by_grade.png` | Figure | Default rate per loan grade A–G |
| `roc_curves.png` | Figure | Overlaid ROC curves for both models |
| `feature_importance.png` | Figure | Top-12 Random Forest feature importances |
| `score_distribution.png` | Figure | PD histogram split by actual outcome |
| `business_analytics.png` | Figure | Profit curve, approval rate curve, EL by tier |

---

## Project structure

```
credit_risk_project/
│
├── credit_risk_pipeline_final.py   ← single-file pipeline
│
├── src/                            ← modular version (same logic, split by stage)
│   ├── data_generation.py          Stage 1 — data
│   ├── preprocessing.py            Stage 2 — cleaning & features
│   ├── eda.py                      Stage 3 — EDA plots
│   ├── modeling.py                 Stage 4 — training & evaluation
│   ├── risk_scoring.py             Stage 5 — PD scoring & tiers
│   ├── business_layer.py           Stage 6 — EL & threshold sweep
│   └── run_pipeline.py             CLI runner for modular version
│
├── notebooks/
│   └── credit_risk_pipeline.ipynb  ← fully executed Jupyter notebook
│
├── tests/
│   ├── test_preprocessing.py       7 assertions
│   ├── test_modeling.py            6 assertions
│   ├── test_business_layer.py      8 assertions
│   └── test_data_loader.py         5 assertions (real CSV quirks)
│
├── outputs/                        ← generated at runtime (gitignored)
├── assets/                         ← pre-generated figures (committed)
├── data/                           ← place your Lending Club CSV here
│   └── .gitkeep
│
├── docs/
│   ├── methodology.md              detailed methodology write-up
│   └── getting_the_data.md        step-by-step download instructions
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Running tests

Tests use `pytest` and cover the three core modules:

```bash
pip install pytest
pytest tests/ -v
```

```
tests/test_preprocessing.py   7 passed
tests/test_modeling.py        6 passed
tests/test_business_layer.py  8 passed
tests/test_data_loader.py     5 passed
─────────────────────────────────────
26 passed
```

---

## Methodology

Key decisions:

- **Class imbalance** — upsampling (not SMOTE) to keep the training distribution simple and reproducible; test set untouched
- **Scaling** — `StandardScaler` applied only to Logistic Regression inputs; Random Forest is scale-invariant
- **Evaluation metric** — ROC-AUC (primary) and Average Precision (secondary); accuracy is misleading under heavy class imbalance
- **Recovery rate** — fixed at 40%; in production this should vary by loan type, collateral, and economic cycle
- **No temporal split** — for production deployment, a time-aware walk-forward validation should replace the random split used here

See [`docs/methodology.md`](docs/methodology.md) for the full write-up.

---

## Requirements

```
numpy
pandas
scikit-learn
matplotlib
seaborn
kaggle
pytest
```

Install all:
```bash
pip install -r requirements.txt
```

---

