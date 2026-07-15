"""
ml_service.py — Singleton ML model loader and prediction service.
Loads the saved model bundle once at startup and exposes clean
prediction methods used by all Django views.
"""

import os
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from django.conf import settings

logger = logging.getLogger(__name__)

# ─── Module-level singleton (loaded once at first call) ───────────────────────
_model_bundle: dict | None = None


def _load_model() -> dict:
    """Load model.pkl from the path defined in Django settings."""
    global _model_bundle
    if _model_bundle is not None:
        return _model_bundle

    model_path = str(settings.MODEL_PATH)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"model.pkl not found at '{model_path}'. "
            "Please run `python train.py` first to generate it."
        )
    _model_bundle = joblib.load(model_path)
    logger.info("Loaded ML model: %s", _model_bundle.get("model_name", "unknown"))
    return _model_bundle


# ─── Feature columns must match the order used during training ────────────────
# These are the base continuous/binary columns; the income_type dummies are
# added dynamically in _build_feature_row().
BASE_FEATURE_COLS = [
    "age",
    "annual_income",
    "employment_duration",
    "num_dependents",
    "existing_loans",
    "credit_score",
    "debt_to_income_ratio",
    "education_encoded",
    "owns_property",
    "owns_car",
    "months_since_default",
    "has_prior_default",
    "income_employed",
    "income_pensioner",
    "income_self_employed",
    "income_student",
]

EDUCATION_MAP = {"secondary": 1, "bachelor": 2, "master": 3, "phd": 4}

HIGH_RISK_CODES = {"2", "3"}


def _build_feature_row(data: dict) -> pd.DataFrame:
    """
    Convert a raw input dict (from a form or JSON POST) into a single-row
    DataFrame with the exact feature columns the trained pipeline expects.
    """
    # ── Categorical → numeric ──────────────────────────────────────────────
    income_type = data.get("income_type", "employed")
    row = {
        "age":                  float(data.get("age", 30)),
        "annual_income":        float(data.get("annual_income", 50000)),
        "employment_duration":  float(data.get("employment_duration", 3)),
        "num_dependents":       int(data.get("num_dependents", 0)),
        "existing_loans":       int(data.get("existing_loans", 0)),
        "credit_score":         float(data.get("credit_score", 650)),
        "debt_to_income_ratio": float(data.get("debt_to_income_ratio", 0.3)),
        "education_encoded":    EDUCATION_MAP.get(
                                    data.get("education_level", "bachelor"), 2),
        "owns_property":        int(data.get("owns_property", 0)),
        "owns_car":             int(data.get("owns_car", 0)),
        # months_since_default: 0 if no prior default
        "months_since_default": float(data.get("months_since_default", 0)),
        "has_prior_default":    int(data.get("has_prior_default", 0)),
        # One-hot income type
        "income_employed":      int(income_type == "employed"),
        "income_pensioner":     int(income_type == "pensioner"),
        "income_self_employed": int(income_type == "self_employed"),
        "income_student":       int(income_type == "student"),
    }
    return pd.DataFrame([row])


def predict_single(data: dict) -> dict[str, Any]:
    """
    Predict approval for a single applicant.

    Args:
        data: dict of raw form/JSON fields

    Returns:
        {
            "prediction":       1 | 0,
            "label":            "Approved" | "High-Risk",
            "probability":      0.0 – 1.0,
            "model_name":       str,
            "risk_level":       "low" | "medium" | "high",
        }
    """
    bundle    = _load_model()
    pipeline  = bundle["model"]
    X         = _build_feature_row(data)

    # Align columns with trained feature set
    trained_cols = bundle.get("feature_cols", BASE_FEATURE_COLS)
    for col in trained_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[trained_cols]

    prediction  = int(pipeline.predict(X)[0])
    probability = float(pipeline.predict_proba(X)[0][1])  # P(approved)

    # Map probability to a human-readable risk tier
    if probability >= 0.75:
        risk_level = "low"
    elif probability >= 0.45:
        risk_level = "medium"
    else:
        risk_level = "high"

    return {
        "prediction":  prediction,
        "label":       "Approved" if prediction == 1 else "High-Risk",
        "probability": round(probability, 4),
        "model_name":  bundle.get("model_name", "Unknown"),
        "risk_level":  risk_level,
    }


def predict_batch(records: list[dict]) -> list[dict[str, Any]]:
    """
    Predict approval for a list of applicants (Compliance Dashboard).

    Args:
        records: list of applicant dicts

    Returns:
        list of result dicts with the same keys as predict_single(), plus
        "applicant_id" if provided in the input.
    """
    results = []
    for idx, record in enumerate(records):
        result = predict_single(record)
        result["applicant_id"] = record.get("applicant_id", f"APP-{idx + 1:04d}")
        result["name"]         = record.get("name", f"Applicant {idx + 1}")
        results.append(result)
    return results


def get_model_info() -> dict:
    """Return metadata about the currently loaded model."""
    bundle = _load_model()
    return {
        "model_name": bundle.get("model_name", "Unknown"),
        "metrics":    bundle.get("metrics", {}),
    }
