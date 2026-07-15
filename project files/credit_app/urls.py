"""
credit_app/urls.py — Root URL configuration
"""

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    # Django admin (optional, useful for debugging)
    path("admin/", admin.site.urls),

    # All predictor app routes live under the root path
    path("", include("predictor.urls")),
]
