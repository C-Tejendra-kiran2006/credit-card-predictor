"""
views.py — Django views for all three portal scenarios.

Routes:
    GET  /                          → index / home
    GET  /analyst/                  → Scenario 1: Bank Analyst Portal (form)
    POST /analyst/predict/          → Scenario 1: single prediction (JSON)
    GET  /compliance/               → Scenario 2: Compliance Dashboard
    POST /compliance/batch-screen/  → Scenario 2: batch prediction (JSON)
    GET  /customer/                 → Scenario 3: Customer Self-Service
    POST /customer/check/           → Scenario 3: eligibility check (JSON)
    GET  /api/model-info/           → model metadata endpoint
"""

import json
import logging

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .ml_service import predict_single, predict_batch, get_model_info

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────────────────────────────────────

def index(request):
    """Landing page — links to all three portals."""
    try:
        model_info = get_model_info()
    except FileNotFoundError:
        model_info = {"model_name": "Not trained yet — run train.py", "metrics": {}}
    return render(request, "index.html", {"model_info": model_info})


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 1 — BANK ANALYST PORTAL
# ─────────────────────────────────────────────────────────────────────────────

def analyst_portal(request):
    """Render the Bank Analyst Portal page."""
    return render(request, "analyst_portal.html")


@csrf_exempt
@require_http_methods(["POST"])
def analyst_predict(request):
    """
    POST /analyst/predict/
    Accept a single applicant's data as JSON and return an approval decision.

    Request body (JSON):
        { age, annual_income, employment_duration, credit_score, ... }

    Response (JSON):
        { prediction, label, probability, model_name, risk_level }
    """
    try:
        data   = json.loads(request.body)
        result = predict_single(data)
        logger.info("Analyst prediction: %s (p=%.4f)", result["label"], result["probability"])
        return JsonResponse({"success": True, **result})

    except FileNotFoundError as exc:
        return JsonResponse({"success": False, "error": str(exc)}, status=503)
    except (ValueError, KeyError) as exc:
        return JsonResponse({"success": False, "error": f"Invalid input: {exc}"}, status=400)
    except Exception as exc:
        logger.exception("Unexpected error in analyst_predict")
        return JsonResponse({"success": False, "error": "Internal server error."}, status=500)


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 2 — COMPLIANCE DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def compliance_dashboard(request):
    """Render the Compliance Dashboard page."""
    return render(request, "compliance_dashboard.html")


@csrf_exempt
@require_http_methods(["POST"])
def compliance_batch_screen(request):
    """
    POST /compliance/batch-screen/
    Accept a list of applicant records and return predictions for all of them.

    Request body (JSON):
        { "applicants": [ { ...fields... }, ... ] }

    Response (JSON):
        {
            "success": true,
            "total":   N,
            "approved": N,
            "high_risk": N,
            "results": [ { applicant_id, name, label, probability, risk_level }, ... ]
        }
    """
    try:
        body       = json.loads(request.body)
        applicants = body.get("applicants", [])

        if not isinstance(applicants, list) or len(applicants) == 0:
            return JsonResponse(
                {"success": False, "error": "Provide a non-empty 'applicants' list."},
                status=400
            )

        results   = predict_batch(applicants)
        n_approved = sum(1 for r in results if r["prediction"] == 1)

        return JsonResponse({
            "success":   True,
            "total":     len(results),
            "approved":  n_approved,
            "high_risk": len(results) - n_approved,
            "results":   results,
        })

    except FileNotFoundError as exc:
        return JsonResponse({"success": False, "error": str(exc)}, status=503)
    except Exception as exc:
        logger.exception("Unexpected error in compliance_batch_screen")
        return JsonResponse({"success": False, "error": "Internal server error."}, status=500)


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 3 — CUSTOMER SELF-SERVICE
# ─────────────────────────────────────────────────────────────────────────────

def customer_portal(request):
    """Render the Customer Self-Service eligibility check page."""
    return render(request, "customer_portal.html")


@csrf_exempt
@require_http_methods(["POST"])
def customer_check(request):
    """
    POST /customer/check/
    Public-facing eligibility check — returns a friendly approval decision.

    Same request/response schema as analyst_predict, but the response
    omits internal model metadata to keep the experience user-friendly.
    """
    try:
        data   = json.loads(request.body)
        result = predict_single(data)

        # Return a customer-friendly response (no internal model details)
        return JsonResponse({
            "success":    True,
            "eligible":   result["prediction"] == 1,
            "label":      result["label"],
            "probability": result["probability"],
            "risk_level": result["risk_level"],
            "message": (
                "Congratulations! Based on your profile, you appear eligible "
                "for a credit card. A bank representative will review your application."
                if result["prediction"] == 1
                else "We're sorry — based on the information provided, your profile "
                     "does not currently meet our approval criteria. "
                     "Consider improving your credit score or reducing existing debt."
            ),
        })

    except FileNotFoundError as exc:
        return JsonResponse({"success": False, "error": str(exc)}, status=503)
    except Exception as exc:
        logger.exception("Unexpected error in customer_check")
        return JsonResponse({"success": False, "error": "Internal server error."}, status=500)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL INFO API
# ─────────────────────────────────────────────────────────────────────────────

def api_model_info(request):
    """GET /api/model-info/ — return model metadata as JSON."""
    try:
        return JsonResponse({"success": True, **get_model_info()})
    except FileNotFoundError as exc:
        return JsonResponse({"success": False, "error": str(exc)}, status=503)
