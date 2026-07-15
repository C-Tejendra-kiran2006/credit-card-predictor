"""
watson_deploy.py — Deploy the saved Credit Card Approval model to
                   IBM Watson Machine Learning (WML) on IBM Cloud.

Pipeline:
    Step 1  Authenticate with IBM Cloud using an API key
    Step 2  Connect to your Watson Machine Learning service instance
    Step 3  Create (or reuse) a deployment space
    Step 4  Upload the model artifact (model.pkl) to the WML repository
    Step 5  Deploy the model as a real-time online scoring endpoint
    Step 6  Run a test prediction against the live endpoint
    Step 7  Print the scoring URL for use in your Django settings

Prerequisites:
    pip install ibm-watson-machine-learning ibm-cloud-sdk-core python-dotenv
    export IBM_CLOUD_API_KEY="your-ibm-cloud-api-key"
    export IBM_WML_INSTANCE_URL="https://us-south.ml.cloud.ibm.com"

Usage:
    python watson_deploy.py

Environment variables (or .env file):
    IBM_CLOUD_API_KEY     — IBM Cloud API key (required)
    IBM_WML_INSTANCE_URL  — WML service URL  (default: us-south)
    WML_SPACE_NAME        — Deployment space name (default: credit-predictor-space)
    WML_MODEL_NAME        — Model name shown in WML UI
    WML_DEPLOYMENT_NAME   — Deployment name shown in WML UI
"""

import os
import sys
import json
import joblib
import logging
from pathlib import Path

from dotenv import load_dotenv

# ── Load environment variables ───────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
IBM_CLOUD_API_KEY    = os.environ.get("IBM_CLOUD_API_KEY", "")
IBM_WML_INSTANCE_URL = os.environ.get("IBM_WML_INSTANCE_URL", "https://us-south.ml.cloud.ibm.com")
WML_SPACE_NAME       = os.environ.get("WML_SPACE_NAME",      "credit-predictor-space")
WML_MODEL_NAME       = os.environ.get("WML_MODEL_NAME",      "CreditCardApprovalModel")
WML_DEPLOYMENT_NAME  = os.environ.get("WML_DEPLOYMENT_NAME", "credit-card-approval-deployment")

MODEL_PATH           = Path(__file__).parent / "model.pkl"
SOFTWARE_SPEC_NAME   = "runtime-23.1-py3.10"   # WML pre-built Python 3.10 runtime


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — Validate prerequisites
# ─────────────────────────────────────────────────────────────────────────────

def validate_prerequisites():
    """Check that all required environment variables and files are present."""
    errors = []

    if not IBM_CLOUD_API_KEY:
        errors.append(
            "IBM_CLOUD_API_KEY is not set. "
            "Generate one at https://cloud.ibm.com/iam/apikeys"
        )

    if not MODEL_PATH.exists():
        errors.append(
            f"model.pkl not found at {MODEL_PATH}. "
            "Run `python train.py` first."
        )

    try:
        from ibm_watson_machine_learning import APIClient  # noqa: F401
    except ImportError:
        errors.append(
            "ibm-watson-machine-learning not installed. "
            "Run: pip install ibm-watson-machine-learning"
        )

    if errors:
        for e in errors:
            log.error("  ✖  %s", e)
        sys.exit(1)

    log.info("Prerequisites validated.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Authenticate & create WML client
# ─────────────────────────────────────────────────────────────────────────────

def create_wml_client():
    """
    Authenticate with IBM Cloud and return a configured APIClient instance.

    IBM WML uses IAM (Identity and Access Management) token-based auth.
    The SDK handles token refresh automatically.
    """
    from ibm_watson_machine_learning import APIClient

    wml_credentials = {
        "apikey": IBM_CLOUD_API_KEY,
        "url":    IBM_WML_INSTANCE_URL,
    }

    log.info("Step 1 — Connecting to WML at %s …", IBM_WML_INSTANCE_URL)
    client = APIClient(wml_credentials)
    log.info("         Connected. Client version: %s", client.version)
    return client


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Create or reuse a Deployment Space
# ─────────────────────────────────────────────────────────────────────────────

def get_or_create_space(client) -> str:
    """
    Deployment spaces act as isolated environments for models & deployments.
    Returns the space_id of an existing or newly created space.
    """
    log.info("Step 2 — Looking for deployment space '%s' …", WML_SPACE_NAME)

    # Search existing spaces
    existing = client.spaces.list(limit=100)
    for space in existing.get("resources", []):
        if space["entity"]["name"] == WML_SPACE_NAME:
            space_id = space["metadata"]["id"]
            log.info("         Found existing space: %s", space_id)
            client.set.default_space(space_id)
            return space_id

    # Create a new space
    log.info("         Space not found — creating …")
    space_meta = {
        client.spaces.ConfigurationMetaNames.NAME:        WML_SPACE_NAME,
        client.spaces.ConfigurationMetaNames.DESCRIPTION: "Credit card approval predictor deployment space",
    }
    space      = client.spaces.store(meta_props=space_meta)
    space_id   = client.spaces.get_id(space)
    log.info("         Created space: %s", space_id)

    client.set.default_space(space_id)
    return space_id


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Resolve the software specification UID
# ─────────────────────────────────────────────────────────────────────────────

def get_software_spec_uid(client) -> str:
    """
    WML requires you to declare which Python runtime your model uses.
    We use the pre-built 'runtime-23.1-py3.10' spec.
    """
    log.info("Step 3 — Resolving software spec '%s' …", SOFTWARE_SPEC_NAME)
    uid = client.software_specifications.get_uid_by_name(SOFTWARE_SPEC_NAME)
    if not uid:
        raise RuntimeError(
            f"Software spec '{SOFTWARE_SPEC_NAME}' not found. "
            "Check available specs: client.software_specifications.list()"
        )
    log.info("         Software spec UID: %s", uid)
    return uid


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Upload the model to the WML repository
# ─────────────────────────────────────────────────────────────────────────────

def upload_model(client, software_spec_uid: str) -> str:
    """
    Store the serialized model bundle in the WML model repository.
    Returns the model_uid needed for deployment.

    WML supports sklearn pipelines natively — the bundle is uploaded as a
    scikit-learn model type, which handles our Pipeline-wrapped classifiers.
    """
    log.info("Step 4 — Uploading model artifact from %s …", MODEL_PATH)

    # Load the bundle to extract metadata
    bundle = joblib.load(str(MODEL_PATH))
    model_name_tag = bundle.get("model_name", "unknown").replace(" ", "_")

    model_meta = {
        client.repository.ModelMetaNames.NAME:               WML_MODEL_NAME,
        client.repository.ModelMetaNames.DESCRIPTION:        (
            f"Credit card approval predictor — best model: {model_name_tag}"
        ),
        client.repository.ModelMetaNames.TYPE:               "scikit-learn_1.1",
        client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:  software_spec_uid,
    }

    # WML uploads the entire file as a binary archive
    stored_model = client.repository.store_model(
        model=str(MODEL_PATH),
        meta_props=model_meta,
    )

    model_uid = client.repository.get_model_id(stored_model)
    log.info("         Model stored. UID: %s", model_uid)
    return model_uid


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Deploy the model as an online (real-time) scoring endpoint
# ─────────────────────────────────────────────────────────────────────────────

def deploy_model(client, model_uid: str) -> tuple[str, str]:
    """
    Create an online deployment for the stored model.
    Returns (deployment_uid, scoring_url).
    """
    log.info("Step 5 — Creating online deployment '%s' …", WML_DEPLOYMENT_NAME)

    deployment_meta = {
        client.deployments.ConfigurationMetaNames.NAME:             WML_DEPLOYMENT_NAME,
        client.deployments.ConfigurationMetaNames.DESCRIPTION:      "Credit card approval real-time scoring",
        client.deployments.ConfigurationMetaNames.ONLINE:           {},   # online = REST endpoint
    }

    deployment     = client.deployments.create(model_uid, meta_props=deployment_meta)
    deployment_uid = client.deployments.get_id(deployment)
    scoring_url    = client.deployments.get_scoring_href(deployment)

    log.info("         Deployment UID: %s", deployment_uid)
    log.info("         Scoring URL:    %s", scoring_url)
    return deployment_uid, scoring_url


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Test prediction against the live WML endpoint
# ─────────────────────────────────────────────────────────────────────────────

def test_deployment(client, deployment_uid: str):
    """
    Send a sample scoring request to verify the endpoint is live.

    WML expects the payload in Watson's InputData format:
        { "input_data": [{ "fields": [...], "values": [[...]] }] }
    """
    log.info("Step 6 — Running test prediction against live endpoint …")

    # Sample applicant: mid-income employed with decent credit
    fields = [
        "age", "annual_income", "employment_duration", "num_dependents",
        "existing_loans", "credit_score", "debt_to_income_ratio",
        "education_encoded", "owns_property", "owns_car",
        "months_since_default", "has_prior_default",
        "income_employed", "income_pensioner", "income_self_employed", "income_student"
    ]
    values = [[35, 65000, 6, 1, 1, 700, 0.28, 2, 1, 1, 0, 0, 1, 0, 0, 0]]

    scoring_payload = {
        "input_data": [{"fields": fields, "values": values}]
    }

    response = client.deployments.score(deployment_uid, scoring_payload)
    log.info("         Test response: %s", json.dumps(response, indent=2))

    prediction = response["predictions"][0]["values"][0]
    log.info("         Test prediction: %s", prediction)
    return response


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Save deployment details for use in Django
# ─────────────────────────────────────────────────────────────────────────────

def save_deployment_info(deployment_uid: str, scoring_url: str):
    """
    Write deployment metadata to watson_deployment_info.json so the
    Django app can read and call the live WML endpoint at runtime.

    In Django settings.py, add:
        import json
        with open("watson_deployment_info.json") as f:
            WATSON_DEPLOYMENT = json.load(f)
        WATSON_SCORING_URL    = WATSON_DEPLOYMENT["scoring_url"]
        WATSON_DEPLOYMENT_UID = WATSON_DEPLOYMENT["deployment_uid"]
        WATSON_API_KEY        = os.environ["IBM_CLOUD_API_KEY"]
    """
    info = {
        "deployment_uid":  deployment_uid,
        "scoring_url":     scoring_url,
        "model_name":      WML_MODEL_NAME,
        "deployment_name": WML_DEPLOYMENT_NAME,
        "wml_instance_url": IBM_WML_INSTANCE_URL,
    }
    out_path = Path(__file__).parent / "watson_deployment_info.json"
    out_path.write_text(json.dumps(info, indent=2))
    log.info("Step 7 — Deployment info saved to %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  IBM WATSON ML — CREDIT CARD MODEL DEPLOYMENT PIPELINE")
    print("=" * 65)

    # Pre-flight checks
    validate_prerequisites()

    # Run deployment pipeline
    client           = create_wml_client()
    space_id         = get_or_create_space(client)          # noqa: F841
    sw_uid           = get_software_spec_uid(client)
    model_uid        = upload_model(client, sw_uid)
    dep_uid, dep_url = deploy_model(client, model_uid)

    # Smoke test
    test_deployment(client, dep_uid)

    # Persist details
    save_deployment_info(dep_uid, dep_url)

    print("\n" + "=" * 65)
    print("  ✔  DEPLOYMENT COMPLETE")
    print(f"     Scoring URL: {dep_url}")
    print("=" * 65)
    print("""
  Next steps:
  ─────────────────────────────────────────────────────────────
  1. Add to your .env:
       IBM_CLOUD_API_KEY=<your-key>

  2. In Django views.py, to call the WML endpoint directly:

       from ibm_watson_machine_learning import APIClient
       import json, os

       with open("watson_deployment_info.json") as f:
           info = json.load(f)

       client = APIClient({"apikey": os.environ["IBM_CLOUD_API_KEY"],
                            "url": info["wml_instance_url"]})
       payload = {"input_data": [{"fields": FIELDS, "values": [values]}]}
       result  = client.deployments.score(info["deployment_uid"], payload)

  3. Or call the scoring_url via HTTP POST with a Bearer token:
       curl -X POST '{scoring_url}' \\
            -H 'Authorization: Bearer <IAM_TOKEN>' \\
            -H 'Content-Type: application/json' \\
            -d '{payload}'
  ─────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
