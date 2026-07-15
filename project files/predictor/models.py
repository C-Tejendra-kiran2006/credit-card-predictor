"""
models.py — Database models for audit logging of predictions.
"""

from django.db import models


class PredictionAuditLog(models.Model):
    """
    Logs every prediction made through any of the three portals.
    Useful for compliance audits and model monitoring.
    """
    PORTAL_CHOICES = [
        ("analyst",    "Bank Analyst Portal"),
        ("compliance", "Compliance Dashboard"),
        ("customer",   "Customer Self-Service"),
    ]

    portal          = models.CharField(max_length=20, choices=PORTAL_CHOICES)
    # Stores the input features as a JSON blob
    input_data      = models.JSONField()
    prediction      = models.IntegerField(help_text="1 = Approved, 0 = High-Risk")
    probability     = models.FloatField(help_text="Probability of approval (0–1)")
    created_at      = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Prediction Audit Log"

    def __str__(self):
        result = "Approved" if self.prediction == 1 else "High-Risk"
        return f"[{self.portal}] {result} — {self.created_at:%Y-%m-%d %H:%M}"
