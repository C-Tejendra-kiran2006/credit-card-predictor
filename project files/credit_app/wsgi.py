"""
wsgi.py — WSGI config for credit_app project.
Used by Gunicorn in production:
    gunicorn credit_app.wsgi:application --bind 0.0.0.0:8000
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "credit_app.settings")
application = get_wsgi_application()
