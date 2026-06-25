"""
credit_risk_pipeline_final.py
═════════════════════════════
Full credit risk modeling pipeline — Lending Club dataset.

Stages
------
  1. Data        — synthetic generator OR real Lending Club CSV loader
  2. Preprocessing — impute, feature-engineer, encode
  3. EDA           — income, interest rate, loan amount, grade plots
  4. Modeling      — Logistic Regression + Random Forest, ROC, feature importance
  5. Risk scoring  — PD score + Low / Medium / High tier assignment
  6. Business      — Expected Loss, revenue estimate, approval threshold sweep
  7. Outputs       — figures, scored CSV, metrics JSON

Usage
-----
  # Synthetic data (no download needed):
  python credit_risk_pipeline_final.py

  # Real Lending Club CSV (see "Getting the data" section at bottom):
  python credit_risk_pipeline_final.py --data path/to/accepted_2007_to_2018Q4.csv

  # Limit rows for a fast test on large files:
  python credit_risk_pipeline_final.py --data path/to/loan.csv --nrows 100000

  # Custom output directory:
  python credit_risk_pipeline_final.py --output results/

Requirements
------------
  pip install numpy pandas scikit-learn matplotlib seaborn

Getting the real Lending Club data
-----------------------------------
  pip install kaggle
  # Place kaggle.json in ~/.kaggle/ (from kaggle.com → Account → API)
  kaggle datasets download -d wordsforthewise/lending-club -p data/ --unzip
  # File: data/accepted_2007_to_2018Q4.csv  (~1.6 GB, 2.26 M rows)

  # Smaller alternative (2007–2015, ~430 MB):
  kaggle datasets download -d adarshsng/lending-club-loan-data-csv -p data/ --unzip
  # File: data/loan.csv
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — DATA COLLECTION
# ══════════════════════════════════════════════════════════════════════════════

EMP_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8,
    "9 years": 9, "10+ years": 10, "Unknown": -1,
}

DEFAULT_STATUSES = {
    "Charged Off", "Default", "Late (31-120 days)", "Late (16-30 days)",
    "Does not meet the credit policy. Status:Charged Off",
}
PAID_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}
REQUIRED_COLS = [
    "loan_amnt", "term", "int_rate", "grade", "emp_length",
    "home_ownership", "annual_inc", "purpose", "dti",
    "delinq_2yrs", "fico_range_low", "open_acc", "pub_rec",
    "revol_util", "loan_status",
]
PCT_STRING_COLS = ["int_rate", "revol_util"]


def generate_dataset(n: int = 15_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic Lending Club–style loan dataset.

    Uses a logistic default-probability model with realistic feature
    correlations. Injects ~3–5% missing values into key columns.
    """
    rng = np.random.default_rng(seed)

    annual_inc     = rng.lognormal(mean=10.8, sigma=0.6, size=n).clip(20_000, 300_000)
    loan_amnt      = rng.lognormal(mean=9.4,  sigma=0.6, size=n).clip(1_000,  40_000)
    dti            = rng.beta(2, 5, n) * 40
    int_rate       = (6 + dti * 0.4 + rng.normal(0, 1.5, n)).clip(5, 30)
    fico_range_low = (rng.beta(5, 3, n) * 400 + 450).clip(450, 850)
    revol_util     = (rng.beta(2, 3, n) * 100).clip(0, 100)
    open_acc       = rng.poisson(11, n).clip(1, 40)
    pub_rec        = rng.choice([0, 1, 2, 3], n, p=[0.80, 0.12, 0.05, 0.03])
    delinq_2yrs    = rng.choice([0, 1, 2, 3], n, p=[0.75, 0.15, 0.07, 0.03])

    emp_keys   = list(EMP_MAP.keys())
    emp_length = rng.choice(
        emp_keys, n,
        p=[0.10, 0.08, 0.10, 0.09, 0.08, 0.10, 0.07, 0.08, 0.07, 0.08, 0.15, 0.00],
    )
    home_ownership = rng.choice(
        ["RENT", "OWN", "MORTGAGE", "OTHER"], n, p=[0.45, 0.15, 0.37, 0.03])
    purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement",
         "other", "major_purchase", "medical"],
        n, p=[0.42, 0.22, 0.13, 0.10, 0.08, 0.05],
    )
    term  = rng.choice(["36 months", "60 months"], n, p=[0.70, 0.30])
    grade = rng.choice(
        ["A", "B", "C", "D", "E", "F", "G"], n,
        p=[0.20, 0.25, 0.22, 0.15, 0.10, 0.05, 0.03],
    )

    log_odds = (
        -3.5
        + 0.05 * (int_rate - 12)
        + 0.02 * dti
        - 0.003 * (fico_range_low - 650)
        + 0.01  * (revol_util - 50)
        + 0.3   * pub_rec
        + 0.2   * delinq_2yrs
        - 0.5   * np.log1p(annual_inc / 10_000)
        + 0.3   * (term == "60 months").astype(float)
        + 0.1   * rng.standard_normal(n)
    )
    p_default   = 1 / (1 + np.exp(-log_odds))
    loan_status = (rng.random(n) < p_default).astype(int)

    for arr, rate in [(dti, 0.03), (revol_util, 0.04), (annual_inc, 0.02)]:
        arr[rng.random(n) < rate] = np.nan
    emp_arr = np.array(emp_length, dtype=object)
    emp_arr[rng.random(n) < 0.05] = None
    emp_length = emp_arr.tolist()

    return pd.DataFrame({
        "loan_amnt": loan_amnt, "term": term, "int_rate": int_rate,
        "grade": grade, "emp_length": emp_length,
        "home_ownership": home_ownership, "annual_inc": annual_inc,
        "purpose": purpose, "dti": dti, "delinq_2yrs": delinq_2yrs,
        "fico_range_low": fico_range_low, "open_acc": open_acc,
        "pub_rec": pub_rec, "revol_util": revol_util,
        "loan_status": loan_status,
    })


def load_from_csv(path: str, nrows: int = None) -> pd.DataFrame:
    """
    Load and clean a real Lending Club CSV export.

    Handles all known quirks of the Kaggle / official LC data files:
      - Junk second header row ("Notes offered by Prospectus…")
      - int_rate and revol_util stored as strings: "10.65%" → 10.65
      - emp_length "n/a" / NaN → "Unknown"
      - loan_status multi-class → binary; transitory statuses dropped
      - Numeric columns stored as mixed-type strings coerced to float
      - Missing fico_range_low estimated from grade
    """
    print(f"  Reading {path} …")

    # Detect junk second row
    probe = pd.read_csv(path, nrows=2, header=0, low_memory=False)
    skip_second = _is_junk_row(probe.iloc[0])

    read_kwargs = dict(header=0, low_memory=False,
                       skiprows=[1] if skip_second else None)
    if nrows:
        read_kwargs["nrows"] = nrows

    df = pd.read_csv(path, **read_kwargs)
    print(f"  Raw shape: {df.shape}")

    df.columns = df.columns.str.strip().str.lower()

    # Strip % from percent-string columns
    for col in PCT_STRING_COLS:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.replace("%", "", regex=False)
                .str.strip().replace("nan", np.nan).astype(float)
            )

    # fico_range_low fallback
    if "fico_range_low" not in df.columns:
        if "fico_range_high" in df.columns:
            df["fico_range_low"] = pd.to_numeric(df["fico_range_high"], errors="coerce") - 4
        else:
            grade_map = {"A": 760, "B": 720, "C": 690, "D": 660, "E": 640, "F": 620, "G": 600}
            df["fico_range_low"] = (
                df["grade"].astype(str).str[0].map(grade_map).fillna(670)
                if "grade" in df.columns else 670
            )

    # Numeric coercion
    for col in ["loan_amnt", "annual_inc", "dti", "delinq_2yrs",
                "fico_range_low", "open_acc", "pub_rec", "revol_util"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalise string columns
    if "emp_length" in df.columns:
        df["emp_length"] = (df["emp_length"].astype(str).str.strip()
                            .replace({"nan": "Unknown", "n/a": "Unknown", "N/A": "Unknown"}))
    if "term" in df.columns:
        df["term"] = df["term"].astype(str).str.strip()
    if "grade" in df.columns:
        df["grade"] = df["grade"].astype(str).str.strip().str.upper()
    if "home_ownership" in df.columns:
        df["home_ownership"] = (df["home_ownership"].astype(str).str.strip().str.upper()
                                .replace({"ANY": "OTHER", "NONE": "OTHER"}))

    # Binary loan_status
    if "loan_status" not in df.columns:
        raise ValueError("Column 'loan_status' not found in CSV.")
    df["loan_status"] = df["loan_status"].astype(str).str.strip()
    mapping = {s: 1 for s in DEFAULT_STATUSES}
    mapping.update({s: 0 for s in PAID_STATUSES})
    df["loan_status"] = df["loan_status"].map(mapping)
    df = df.dropna(subset=["loan_status"])
    df["loan_status"] = df["loan_status"].astype(int)

    available = [c for c in REQUIRED_COLS if c in df.columns]
    missing   = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"  Warning: columns not in CSV (will be NaN): {missing}")

    result = df[available].copy()
    for col in missing:
        result[col] = np.nan

    print(f"  Cleaned shape: {result.shape}")
    print(f"  Default rate : {result['loan_status'].mean():.1%}")
    return result


def _is_junk_row(row: pd.Series) -> bool:
    v = str(row.iloc[0]).lower()
    return "notes" in v or "offered" in v or v == "nan"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

FEATURES = [
    "loan_amnt", "term_num", "int_rate", "grade_num", "emp_length_num",
    "home_enc", "annual_inc", "purpose_enc", "dti", "delinq_2yrs",
    "fico_range_low", "open_acc", "pub_rec", "revol_util",
    "inc_to_loan", "monthly_payment_est",
]
TARGET = "loan_status"


def preprocess(df: pd.DataFrame) -> tuple:
    """
    Clean raw data and return (X, y).

    Steps: impute → engineer features → encode categoricals → median-impute residuals.
    """
    df = df.copy()

    # Impute
    for col in ["annual_inc", "dti", "revol_util"]:
        df[col] = df[col].fillna(df[col].median())
    df["emp_length"] = df["emp_length"].fillna("Unknown")

    # Feature engineering
    df["emp_length_num"]      = df["emp_length"].map(lambda x: EMP_MAP.get(str(x), -1))
    df["term_num"]            = df["term"].map({"36 months": 36, "60 months": 60})
    df["grade_num"]           = df["grade"].map({"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7})
    df["inc_to_loan"]         = df["annual_inc"] / df["loan_amnt"]
    df["monthly_payment_est"] = df["loan_amnt"] * (df["int_rate"] / 100) / 12

    # Encoding
    le = LabelEncoder()
    df["home_enc"]    = le.fit_transform(df["home_ownership"].astype(str))
    df["purpose_enc"] = le.fit_transform(df["purpose"].astype(str))

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=FEATURES, index=X.index)

    return X, y


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Summary of missing values per column."""
    miss = df.isnull().sum()
    pct  = (miss / len(df) * 100).round(2)
    return (
        pd.DataFrame({"missing_count": miss, "missing_pct": pct})
        .loc[miss > 0]
        .sort_values("missing_pct", ascending=False)
    )


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — EDA
# ══════════════════════════════════════════════════════════════════════════════

_PALETTE = {0: "#4A90D9", 1: "#E05C5C"}
_LABELS  = {0: "Fully paid", 1: "Default"}


def plot_key_features(df: pd.DataFrame) -> plt.Figure:
    """2×2 EDA grid: income, interest rate, loan amount, default rate by employment."""
    df = df.copy()
    df["status_label"] = df["loan_status"].map(_LABELS)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("EDA — key loan features by default status", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    for s, col in _PALETTE.items():
        sub = df.loc[df["loan_status"] == s, "annual_inc"].clip(0, 200_000)
        ax.hist(sub, bins=40, alpha=0.6, color=col, label=_LABELS[s], density=True)
    ax.set_title("Annual income distribution")
    ax.set_xlabel("Annual income ($)"); ax.set_ylabel("Density"); ax.legend()

    ax = axes[0, 1]
    df.boxplot(column="int_rate", by="status_label", ax=ax, patch_artist=True,
               boxprops=dict(facecolor="#B8D4F0"),
               medianprops=dict(color="#E05C5C", linewidth=2))
    ax.set_title("Interest rate by default status"); ax.set_xlabel("")
    plt.sca(ax); plt.title("Interest rate by default status")

    ax = axes[1, 0]
    for s, col in _PALETTE.items():
        sub = df.loc[df["loan_status"] == s, "loan_amnt"]
        ax.hist(sub, bins=40, alpha=0.6, color=col, label=_LABELS[s], density=True)
    ax.set_title("Loan amount distribution")
    ax.set_xlabel("Loan amount ($)"); ax.set_ylabel("Density"); ax.legend()

    ax = axes[1, 1]
    avg_default = df["loan_status"].mean()
    df.groupby("emp_length")["loan_status"].mean().sort_index().plot(
        kind="bar", ax=ax, color="#4A90D9", edgecolor="white")
    ax.set_title("Default rate by employment length")
    ax.set_xlabel("Employment length"); ax.set_ylabel("Default rate")
    ax.tick_params(axis="x", rotation=45)
    ax.axhline(avg_default, color="#E05C5C", linestyle="--", label=f"Avg {avg_default:.1%}")
    ax.legend()

    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, target: str = "loan_status") -> plt.Figure:
    """Lower-triangle correlation heatmap for all model features + target."""
    cols = [f for f in FEATURES if f in df.columns] + [target]
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0,
                annot=True, fmt=".2f", linewidths=0.4, ax=ax, annot_kws={"size": 7})
    ax.set_title("Feature correlation matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_default_rate_by_grade(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of default rate by loan grade A–G."""
    grade_order = ["A", "B", "C", "D", "E", "F", "G"]
    grade_default = df.groupby("grade")["loan_status"].mean().reindex(grade_order).dropna()

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(grade_default)))
    grade_default.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Default rate by loan grade"); ax.set_xlabel("Grade")
    ax.set_ylabel("Default rate"); ax.tick_params(axis="x", rotation=0)
    ax.axhline(df["loan_status"].mean(), color="#333", linestyle="--",
               linewidth=1, label="Overall avg")
    ax.legend()
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — MODELING
# ══════════════════════════════════════════════════════════════════════════════

def split_and_balance(X: pd.DataFrame, y: pd.Series,
                      test_size: float = 0.2, seed: int = 42) -> tuple:
    """
    Stratified train/test split + upsample minority class in training only.

    Returns: X_train_bal, X_test_scaled, X_test_raw, y_train_bal, y_test, scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    train_df = pd.concat([X_train, y_train], axis=1)
    majority    = train_df[train_df[TARGET] == 0]
    minority    = train_df[train_df[TARGET] == 1]
    minority_up = resample(minority, replace=True,
                           n_samples=len(majority), random_state=seed)
    balanced    = pd.concat([majority, minority_up])

    X_train_bal = balanced.drop(columns=TARGET)
    y_train_bal = balanced[TARGET]

    scaler = StandardScaler()
    X_train_bal_s = pd.DataFrame(
        scaler.fit_transform(X_train_bal),
        columns=X_train_bal.columns, index=X_train_bal.index)
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns, index=X_test.index)

    return X_train_bal_s, X_test_scaled, X_test, y_train_bal, y_test, scaler


def train_logistic_regression(X_train, y_train, C: float = 0.5,
                               seed: int = 42) -> LogisticRegression:
    """Fit and return a Logistic Regression model."""
    lr = LogisticRegression(max_iter=500, C=C, random_state=seed)
    lr.fit(X_train, y_train)
    return lr


def train_random_forest(X_train, y_train, n_estimators: int = 200,
                        max_depth: int = 10, seed: int = 42) -> RandomForestClassifier:
    """Fit and return a Random Forest model."""
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                min_samples_leaf=20, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf


def evaluate_model(model, X_test, y_test, name: str = "Model") -> dict:
    """Print classification report and return metrics dict."""
    proba = model.predict_proba(X_test)[:, 1]
    pred  = model.predict(X_test)
    auc   = roc_auc_score(y_test, proba)
    ap    = average_precision_score(y_test, proba)

    print(f"\n{'─'*50}")
    print(f"{name}  |  AUC={auc:.4f}  |  Avg Precision={ap:.4f}")
    print("─" * 50)
    print(classification_report(y_test, pred, target_names=["Paid", "Default"]))

    return {"name": name, "auc": auc, "ap": ap, "proba": proba, "pred": pred}


def plot_roc_curves(metrics_list: list, y_test: pd.Series) -> plt.Figure:
    """Overlay ROC curves for all evaluated models."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in metrics_list:
        fpr, tpr, _ = roc_curve(y_test, m["proba"])
        ax.plot(fpr, tpr, lw=2, label=f"{m['name']} (AUC={m['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves"); ax.legend()
    plt.tight_layout()
    return fig


def plot_feature_importance(rf: RandomForestClassifier,
                             feature_names: list, top_n: int = 12) -> plt.Figure:
    """Horizontal bar chart of Random Forest feature importances."""
    fi = pd.Series(rf.feature_importances_, index=feature_names).sort_values().tail(top_n)
    fig, ax = plt.subplots(figsize=(7, 5))
    fi.plot(kind="barh", ax=ax, color="#4A90D9")
    ax.set_title(f"Random Forest — feature importance (top {top_n})")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — RISK SCORING
# ══════════════════════════════════════════════════════════════════════════════

RISK_THRESHOLDS = {"Low": 0.15, "Medium": 0.35}


def assign_risk_tier(prob: float, thresholds: dict = RISK_THRESHOLDS) -> str:
    """Map a default probability to Low / Medium / High."""
    if prob < thresholds["Low"]:    return "Low"
    elif prob < thresholds["Medium"]: return "Medium"
    return "High"


def score_loans(X_test: pd.DataFrame, y_test: pd.Series,
                model, original_df: pd.DataFrame) -> pd.DataFrame:
    """Score test-set loans with PD, risk tier, and raw context columns."""
    scored = X_test.copy()
    scored["actual_default"] = y_test.values
    scored["default_prob"]   = model.predict_proba(X_test)[:, 1]

    for col in ["loan_amnt", "int_rate", "annual_inc", "term_num"]:
        if col in original_df.columns:
            scored[col] = original_df.loc[X_test.index, col].values

    scored["risk_tier"] = scored["default_prob"].apply(assign_risk_tier)
    return scored


def risk_tier_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Aggregate count, avg loan, avg PD, and actual default rate per tier."""
    tiers = [t for t in ["Low", "Medium", "High"] if t in scored["risk_tier"].values]
    return (
        scored.groupby("risk_tier")
        .agg(Count=("loan_amnt", "count"), Avg_Loan=("loan_amnt", "mean"),
             Avg_PD=("default_prob", "mean"),
             Actual_Default_Rate=("actual_default", "mean"))
        .round(3).loc[tiers]
    )


def plot_score_distribution(scored: pd.DataFrame) -> plt.Figure:
    """PD score histogram split by actual outcome."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, val, color in [("Fully paid", 0, "#4A90D9"), ("Default", 1, "#E05C5C")]:
        sub = scored.loc[scored["actual_default"] == val, "default_prob"]
        ax.hist(sub, bins=40, alpha=0.6, color=color, label=label, density=True)
    ax.set_xlabel("Predicted default probability"); ax.set_ylabel("Density")
    ax.set_title("Score distribution by actual outcome"); ax.legend()
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — BUSINESS LAYER
# ══════════════════════════════════════════════════════════════════════════════

RECOVERY_RATE = 0.40


def compute_expected_loss(scored: pd.DataFrame,
                           recovery_rate: float = RECOVERY_RATE) -> pd.DataFrame:
    """
    Add EL (Expected Loss) and revenue estimate columns.

    EL = PD × LGD × loan_amnt
    Revenue ≈ loan_amnt × (int_rate/100) × (term_months/12) × (1 − PD)
    """
    df = scored.copy()
    df["LGD"] = 1 - recovery_rate
    df["EL"]  = df["default_prob"] * df["LGD"] * df["loan_amnt"]

    term_months = df.get("term_num", pd.Series(36, index=df.index)).fillna(36)
    df["revenue_est"] = (
        df["loan_amnt"] * (df["int_rate"] / 100)
        * (term_months / 12) * (1 - df["default_prob"])
    )
    return df


def sweep_thresholds(df: pd.DataFrame, n_steps: int = 50,
                     low: float = 0.05, high: float = 0.60) -> pd.DataFrame:
    """
    Sweep approval thresholds and evaluate net profit at each cutoff.

    Returns DataFrame with: threshold, n_approved, approval_rate,
    n_defaults, total_el, total_revenue, net_profit
    """
    n_total = len(df)
    rows = []
    for t in np.linspace(low, high, n_steps):
        approved = df[df["default_prob"] < t]
        rows.append({
            "threshold":     round(t, 4),
            "n_approved":    len(approved),
            "approval_rate": len(approved) / n_total,
            "n_defaults":    int(approved["actual_default"].sum()),
            "total_el":      approved["EL"].sum(),
            "total_revenue": approved["revenue_est"].sum(),
            "net_profit":    approved["revenue_est"].sum() - approved["EL"].sum(),
        })
    return pd.DataFrame(rows)


def optimal_threshold(sweep_df: pd.DataFrame) -> pd.Series:
    """Return the row maximising net profit."""
    return sweep_df.loc[sweep_df["net_profit"].idxmax()]


def tier_business_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate EL, revenue, and profit by risk tier."""
    tiers = [t for t in ["Low", "Medium", "High"] if t in df["risk_tier"].values]
    return (
        df.groupby("risk_tier")
        .agg(Count=("loan_amnt", "count"), Avg_Loan=("loan_amnt", "mean"),
             Avg_PD=("default_prob", "mean"), Avg_EL=("EL", "mean"),
             Total_EL=("EL", "sum"), Est_Revenue=("revenue_est", "sum"))
        .assign(Net_Profit=lambda x: x["Est_Revenue"] - x["Total_EL"])
        .round(2).loc[tiers]
    )


def plot_business_analytics(sweep_df: pd.DataFrame,
                              tier_summary: pd.DataFrame) -> plt.Figure:
    """Three-panel: profit curve, approval rate curve, EL by tier."""
    best = optimal_threshold(sweep_df)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Business analytics", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(sweep_df["threshold"], sweep_df["net_profit"] / 1e6, color="#4A90D9", lw=2)
    ax.axvline(best["threshold"], color="#E05C5C", linestyle="--",
               label=f"Optimal={best['threshold']:.2f}")
    ax.set_xlabel("Approval threshold"); ax.set_ylabel("Net profit ($M)")
    ax.set_title("Profit vs approval threshold"); ax.legend()

    ax = axes[1]
    ax.plot(sweep_df["threshold"], sweep_df["approval_rate"], color="#2ECC71", lw=2)
    ax.axvline(best["threshold"], color="#E05C5C", linestyle="--")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Approval rate")
    ax.set_title("Approval rate vs threshold")

    ax = axes[2]
    tier_order = [t for t in ["Low", "Medium", "High"] if t in tier_summary.index]
    colors     = {"Low": "#2ECC71", "Medium": "#F39C12", "High": "#E05C5C"}
    tier_el    = [tier_summary.loc[t, "Total_EL"] / 1e3 for t in tier_order]
    ax.bar(tier_order, tier_el, color=[colors[t] for t in tier_order])
    ax.set_xlabel("Risk tier"); ax.set_ylabel("Total expected loss ($K)")
    ax.set_title("Expected loss by risk tier")

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — MAIN / CLI RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def _save_fig(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=130, bbox_inches="tight")
    fig.clf()
    plt.close("all")
    print(f"  Saved → {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Credit Risk Pipeline — Lending Club",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data",   type=str, default=None,
                        help="Path to real Lending Club CSV (omit for synthetic data)")
    parser.add_argument("--nrows",  type=int, default=None,
                        help="Limit rows read — useful for large files")
    parser.add_argument("--output", type=str, default="outputs",
                        help="Output directory (default: outputs/)")
    parser.add_argument("--n",      type=int, default=15_000,
                        help="Synthetic dataset size (ignored when --data is set)")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    sns.set_style("whitegrid")

    # ── Stage 1 ──────────────────────────────────────────────────────────
    print("\n[1/6] Data collection")
    if args.data:
        raw_df = load_from_csv(args.data, nrows=args.nrows)
    else:
        print(f"  Generating synthetic dataset (n={args.n:,}, seed={args.seed})")
        raw_df = generate_dataset(n=args.n, seed=args.seed)
    print(f"  Shape: {raw_df.shape}  |  Default rate: {raw_df['loan_status'].mean():.1%}")

    # ── Stage 2 ──────────────────────────────────────────────────────────
    print("\n[2/6] Preprocessing")
    miss = missing_value_report(raw_df)
    if not miss.empty:
        print("  Missing before cleaning:\n", miss.to_string())
    X, y = preprocess(raw_df)
    print(f"  Features: {X.shape[1]}  |  Residual NaNs: {X.isnull().sum().sum()}")

    # ── Stage 3 ──────────────────────────────────────────────────────────
    print("\n[3/6] EDA")
    _save_fig(plot_key_features(raw_df),
              os.path.join(args.output, "eda_overview.png"))
    _save_fig(plot_correlation_heatmap(pd.concat([X, y], axis=1)),
              os.path.join(args.output, "correlation_heatmap.png"))
    _save_fig(plot_default_rate_by_grade(raw_df),
              os.path.join(args.output, "default_rate_by_grade.png"))

    # ── Stage 4 ──────────────────────────────────────────────────────────
    print("\n[4/6] Modeling")
    X_tr, X_te_s, X_te_r, y_tr, y_te, scaler = split_and_balance(X, y, seed=args.seed)

    lr = train_logistic_regression(X_tr, y_tr, seed=args.seed)
    rf = train_random_forest(X_tr, y_tr, seed=args.seed)

    lr_m = evaluate_model(lr, X_te_s, y_te, name="Logistic Regression")
    rf_m = evaluate_model(rf, X_te_r, y_te, name="Random Forest")

    _save_fig(plot_roc_curves([lr_m, rf_m], y_te),
              os.path.join(args.output, "roc_curves.png"))
    _save_fig(plot_feature_importance(rf, FEATURES),
              os.path.join(args.output, "feature_importance.png"))

    # ── Stage 5 ──────────────────────────────────────────────────────────
    print("\n[5/6] Risk scoring")
    scored = score_loans(X_te_r, y_te, rf, raw_df)
    print(risk_tier_summary(scored).to_string())
    _save_fig(plot_score_distribution(scored),
              os.path.join(args.output, "score_distribution.png"))

    # ── Stage 6 ──────────────────────────────────────────────────────────
    print("\n[6/6] Business layer")
    scored_el = compute_expected_loss(scored)
    sweep_df  = sweep_thresholds(scored_el)
    best      = optimal_threshold(sweep_df)
    tier_biz  = tier_business_summary(scored_el)

    print(f"  Optimal threshold : {best['threshold']:.2f}")
    print(f"  Approval rate     : {best['approval_rate']:.1%}")
    print(f"  Net profit (est.) : ${best['net_profit']:,.0f}")
    print(tier_biz.to_string())

    _save_fig(plot_business_analytics(sweep_df, tier_biz),
              os.path.join(args.output, "business_analytics.png"))

    # ── Save artefacts ────────────────────────────────────────────────────
    scored_el.head(500).to_csv(
        os.path.join(args.output, "scored_loans.csv"), index=False)
    sweep_df.to_csv(
        os.path.join(args.output, "threshold_sweep.csv"), index=False)

    metrics = {
        "data_source":          os.path.basename(args.data) if args.data else "synthetic",
        "nrows_used":           args.nrows,
        "logistic_regression":  {"auc": round(lr_m["auc"], 4)},
        "random_forest":        {"auc": round(rf_m["auc"], 4)},
        "optimal_threshold":    round(float(best["threshold"]), 4),
        "optimal_approval_rate":round(float(best["approval_rate"]), 4),
        "estimated_net_profit": round(float(best["net_profit"]), 2),
        "dataset_size":         int(len(raw_df)),
        "default_rate":         round(float(y.mean()), 4),
        "test_size":            int(len(y_te)),
        "risk_tier_counts":     scored["risk_tier"].value_counts().to_dict(),
        "tier_avg_pd":          scored.groupby("risk_tier")["default_prob"].mean().round(4).to_dict(),
        "tier_total_el":        scored_el.groupby("risk_tier")["EL"].sum().round(2).to_dict(),
    }
    with open(os.path.join(args.output, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAll outputs written to: {args.output}/")
    print("Pipeline complete ✓")


if __name__ == "__main__":
    main()
