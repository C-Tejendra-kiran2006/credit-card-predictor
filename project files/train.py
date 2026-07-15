"""
train.py — ML Pipeline for Credit Card Approval Predictor
==========================================================
Steps:
  1. Generate a realistic mock dataset of credit card applicants
  2. Feature engineering (multi-class → binary labels, encoding, scaling)
  3. Handle missing values
  4. Train 4 classifiers: Logistic Regression, Random Forest, XGBoost, Decision Tree
  5. Evaluate all models and select the best by F1-score (macro)
  6. Save the best model to model.pkl

Usage:
    python train.py
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — MOCK DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_mock_dataset(n_samples: int = 3000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic mock dataset for credit card applicants.

    Features generated:
        - age                   : applicant age in years
        - annual_income         : gross annual income (USD)
        - employment_duration   : years at current employer
        - num_dependents        : number of financial dependents
        - existing_loans        : number of active loans
        - credit_score          : numerical credit score (300–850)
        - debt_to_income_ratio  : total monthly debt / gross monthly income
        - income_type           : categorical — employed, self-employed, pensioner, student
        - education_level       : categorical — secondary, bachelor, master, phd
        - owns_property         : binary flag (1 = yes)
        - owns_car              : binary flag (1 = yes)
        - months_since_default  : months since last default (NaN if never)
        - payment_status_code   : multi-class code (C, X, 0, 1, 2, 3) per credit-bureau convention
        - approved              : TARGET — binary (1 = approved, 0 = high-risk / rejected)
    """
    rng = np.random.default_rng(random_seed)

    age                 = rng.integers(21, 70, n_samples)
    annual_income       = rng.lognormal(mean=11.0, sigma=0.6, size=n_samples).round(2)
    employment_duration = np.clip(rng.exponential(scale=4.5, size=n_samples), 0, 40).round(1)
    num_dependents      = rng.integers(0, 5, n_samples)
    existing_loans      = rng.integers(0, 6, n_samples)

    # Credit score correlated loosely with income (kept as float so NaN can be assigned later)
    credit_score_raw    = 0.0003 * annual_income + rng.normal(0, 80, n_samples)
    credit_score        = np.clip(credit_score_raw, 300, 850).astype(float)

    # Debt-to-income ratio (realistic 0.05 – 0.85)
    dti                 = np.clip(rng.beta(a=2, b=5, size=n_samples), 0.05, 0.85).round(4)

    # Categorical features
    income_types        = rng.choice(
        ["employed", "self_employed", "pensioner", "student"],
        n_samples, p=[0.60, 0.20, 0.15, 0.05]
    )
    education_levels    = rng.choice(
        ["secondary", "bachelor", "master", "phd"],
        n_samples, p=[0.25, 0.45, 0.22, 0.08]
    )
    owns_property       = rng.choice([0, 1], n_samples, p=[0.40, 0.60])
    owns_car            = rng.choice([0, 1], n_samples, p=[0.45, 0.55])

    # Months since last default — ~30 % of applicants have a prior default
    has_default         = rng.random(n_samples) < 0.30
    months_since_default = np.where(
        has_default,
        rng.integers(1, 120, n_samples).astype(float),
        np.nan
    )

    # ── Multi-class payment status codes ──────────────────────────────────────
    # Convention mirrors credit-bureau codes:
    #   C  = paid in full / no balance
    #   X  = no payment history
    #   0  = current (on time)
    #   1  = 30-days late
    #   2  = 60-days late
    #   3  = 90+ days late / charge-off
    payment_status_codes = rng.choice(
        ["C", "X", "0", "1", "2", "3"],
        n_samples,
        p=[0.30, 0.10, 0.25, 0.15, 0.12, 0.08]
    )

    # ── Introduce realistic missing values (~5 %) ─────────────────────────────
    for col_vals in [employment_duration, credit_score, dti]:
        missing_mask = rng.random(n_samples) < 0.05
        col_vals[missing_mask] = np.nan  # type: ignore[index]

    df = pd.DataFrame({
        "age":                  age,
        "annual_income":        annual_income,
        "employment_duration":  employment_duration,
        "num_dependents":       num_dependents,
        "existing_loans":       existing_loans,
        "credit_score":         credit_score.astype(float),
        "debt_to_income_ratio": dti,
        "income_type":          income_types,
        "education_level":      education_levels,
        "owns_property":        owns_property,
        "owns_car":             owns_car,
        "months_since_default": months_since_default,
        "payment_status_code":  payment_status_codes,
    })

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

# Payment codes considered "high-risk" — maps to label 0 (rejected)
HIGH_RISK_CODES = {"2", "3"}

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.

    Key transformation — multi-class payment status → binary label:
        C, X, 0, 1  →  approved (1)
        2, 3        →  high-risk / rejected (0)
    """
    df = df.copy()

    # ── 1. Binary target from multi-class payment status code ─────────────────
    df["approved"] = df["payment_status_code"].apply(
        lambda code: 0 if code in HIGH_RISK_CODES else 1
    )

    # ── 2. Ordinal-encode education ───────────────────────────────────────────
    edu_order = {"secondary": 1, "bachelor": 2, "master": 3, "phd": 4}
    df["education_encoded"] = df["education_level"].map(edu_order)

    # ── 3. One-hot encode income_type ─────────────────────────────────────────
    df = pd.get_dummies(df, columns=["income_type"], prefix="income", drop_first=False)

    # ── 4. Binary flag: applicant has ever defaulted ──────────────────────────
    df["has_prior_default"] = df["months_since_default"].notna().astype(int)

    # ── 5. Fill numeric missing values with median (robust to skew) ───────────
    numeric_cols = [
        "employment_duration", "credit_score",
        "debt_to_income_ratio", "months_since_default"
    ]
    for col in numeric_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # ── 6. Drop raw categorical columns no longer needed ─────────────────────
    df.drop(columns=["payment_status_code", "education_level"], inplace=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def build_classifiers() -> dict:
    """
    Return a dict of named sklearn-compatible classifiers.
    Each classifier is wrapped in a Pipeline that:
        1. Imputes remaining NaN with median
        2. Scales features to zero-mean unit-variance
        3. Applies the classifier
    """
    imputer  = SimpleImputer(strategy="median")
    scaler   = StandardScaler()

    classifiers = {
        "Logistic Regression": Pipeline([
            ("impute", imputer),
            ("scale",  scaler),
            ("clf",    LogisticRegression(max_iter=1000, class_weight="balanced",
                                          random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("impute", imputer),
            ("scale",  scaler),
            ("clf",    RandomForestClassifier(n_estimators=200, max_depth=10,
                                              class_weight="balanced", random_state=42,
                                              n_jobs=-1)),
        ]),
        "XGBoost (Gradient Boosting)": Pipeline([
            ("impute", imputer),
            ("scale",  scaler),
            ("clf",    GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
                                                  max_depth=5, random_state=42)),
        ]),
        "Decision Tree": Pipeline([
            ("impute", imputer),
            ("scale",  scaler),
            ("clf",    DecisionTreeClassifier(max_depth=8, class_weight="balanced",
                                              random_state=42)),
        ]),
    }
    return classifiers


def evaluate_models(classifiers: dict, X_train, X_test, y_train, y_test) -> dict:
    """
    Train each classifier, evaluate on the held-out test set,
    and return a dict of results keyed by model name.
    """
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "="*65)
    print("  MODEL EVALUATION RESULTS")
    print("="*65)

    for name, pipeline in classifiers.items():
        # ── Train ──────────────────────────────────────────────────────────
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # ── Metrics ────────────────────────────────────────────────────────
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average="macro")
        roc    = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
        cv_f1  = cross_val_score(pipeline, X_train, y_train,
                                  cv=cv, scoring="f1_macro", n_jobs=-1).mean()

        results[name] = {
            "pipeline":  pipeline,
            "accuracy":  acc,
            "f1_macro":  f1,
            "roc_auc":   roc,
            "cv_f1":     cv_f1,
        }

        print(f"\n  ► {name}")
        print(f"    Accuracy : {acc:.4f}")
        print(f"    F1 Macro : {f1:.4f}  (CV mean: {cv_f1:.4f})")
        print(f"    ROC-AUC  : {roc:.4f}")
        report = classification_report(y_test, y_pred,
                                        target_names=["High-Risk", "Approved"])
        # indent each line by 4 spaces for readability
        print("\n".join("    " + line for line in report.splitlines()))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — MAIN PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  CREDIT CARD APPROVAL PREDICTOR — TRAINING PIPELINE")
    print("=" * 65)

    # 1. Generate dataset ─────────────────────────────────────────────────────
    print("\n[1/5] Generating mock dataset (3 000 applicants)…")
    raw_df = generate_mock_dataset(n_samples=3000)
    print(f"      Dataset shape: {raw_df.shape}")
    print(f"      Missing value summary:\n{raw_df.isnull().sum()[raw_df.isnull().sum() > 0]}")

    # 2. Feature engineering ──────────────────────────────────────────────────
    print("\n[2/5] Engineering features…")
    df = engineer_features(raw_df)
    print(f"      Engineered shape: {df.shape}")
    print(f"      Class balance:\n{df['approved'].value_counts(normalize=True).round(3)}")

    # 3. Train / test split ───────────────────────────────────────────────────
    print("\n[3/5] Splitting data (80 % train / 20 % test, stratified)…")
    feature_cols = [c for c in df.columns if c != "approved"]
    X = df[feature_cols]
    y = df["approved"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"      Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # 4. Train & evaluate all models ──────────────────────────────────────────
    print("\n[4/5] Training and evaluating 4 classifiers…")
    classifiers = build_classifiers()
    results     = evaluate_models(classifiers, X_train, X_test, y_train, y_test)

    # 5. Select best model and save ───────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["f1_macro"])
    best_info = results[best_name]

    print("\n" + "="*65)
    print(f"  ✔  BEST MODEL: {best_name}")
    print(f"     F1 Macro : {best_info['f1_macro']:.4f}")
    print(f"     ROC-AUC  : {best_info['roc_auc']:.4f}")
    print("="*65)

    # Persist model + feature column order together
    model_bundle = {
        "model":        best_info["pipeline"],
        "feature_cols": feature_cols,
        "model_name":   best_name,
        "metrics": {
            "accuracy": best_info["accuracy"],
            "f1_macro": best_info["f1_macro"],
            "roc_auc":  best_info["roc_auc"],
        }
    }

    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    joblib.dump(model_bundle, model_path)
    print(f"\n[5/5] Model saved → {model_path}")
    print("      Run `python manage.py runserver` to start the web application.\n")


if __name__ == "__main__":
    main()
