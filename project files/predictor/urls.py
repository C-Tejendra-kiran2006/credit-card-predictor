"""
predictor/urls.py — URL patterns for all three portal scenarios.
"""

from django.urls import path
from . import views

urlpatterns = [
    # ── Home ──────────────────────────────────────────────────────────────
    path("",                            views.index,                   name="index"),

    # ── Scenario 1: Bank Analyst Portal ───────────────────────────────────
    path("analyst/",                    views.analyst_portal,          name="analyst_portal"),
    path("analyst/predict/",            views.analyst_predict,         name="analyst_predict"),

    # ── Scenario 2: Compliance Dashboard ──────────────────────────────────
    path("compliance/",                 views.compliance_dashboard,    name="compliance_dashboard"),
    path("compliance/batch-screen/",    views.compliance_batch_screen, name="compliance_batch_screen"),

    # ── Scenario 3: Customer Self-Service ─────────────────────────────────
    path("customer/",                   views.customer_portal,         name="customer_portal"),
    path("customer/check/",             views.customer_check,          name="customer_check"),

    # ── Model Info API ────────────────────────────────────────────────────
    path("api/model-info/",             views.api_model_info,          name="api_model_info"),
]
