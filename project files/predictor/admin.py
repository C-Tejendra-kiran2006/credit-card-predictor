"""
admin.py — Register models with Django admin for debugging & compliance review.
"""

from django.contrib import admin
from .models import PredictionAuditLog


@admin.register(PredictionAuditLog)
class PredictionAuditLogAdmin(admin.ModelAdmin):
    list_display  = ("portal", "prediction", "probability", "created_at")
    list_filter   = ("portal", "prediction")
    search_fields = ("portal",)
    readonly_fields = ("portal", "input_data", "prediction", "probability", "created_at")
    ordering      = ("-created_at",)
