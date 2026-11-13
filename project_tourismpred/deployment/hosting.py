import os
import sys
import time
import urllib.request
from pathlib import Path

MODEL_PATH = Path("project_tourismpred/deployment/best_tourism_prediction_model_v1.joblib")
APP_PATH = Path("project_tourismpred/deployment/app.py")
REQ_PATH = Path("project_tourismpred/deployment/requirements.txt")
STREAMLIT_APP_URL = os.getenv("STREAMLIT_APP_URL", "").strip()


def verify_deployment_artifacts():
    required_files = [MODEL_PATH, APP_PATH, REQ_PATH]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        print(f"ERROR: Missing deployment artifacts: {missing}")
        sys.exit(1)
    print(f"Verified deployment artifacts: {[str(p) for p in required_files]}")


def trigger_and_verify_endpoint():
    if not STREAMLIT_APP_URL:
        print(
            "STREAMLIT_APP_URL not set. Streamlit Community Cloud auto-deploys "
            "from the updated 'main' branch commit."
        )
        return

    print(f"Pinging live Streamlit deployment at {STREAMLIT_APP_URL} ...")
    time.sleep(15)
    try:
        req = urllib.request.Request(
            STREAMLIT_APP_URL,
            headers={"User-Agent": "GitHub-Actions-MLOps-Deploy-Check"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            print(f"Live app responded with HTTP {resp.status}")
            if resp.status >= 400:
                sys.exit(1)
    except Exception as exc:
        print(f"Deployment health-check warning: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    verify_deployment_artifacts()
    trigger_and_verify_endpoint()
    print("End-to-end deployment stage completed successfully.")
