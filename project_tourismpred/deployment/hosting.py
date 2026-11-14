import os
import sys
import time
import urllib.request
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("project_tourismpred/deployment/best_tourism_prediction_model_v1.joblib")
APP_PATH = Path("project_tourismpred/deployment/app.py")
REQ_PATH = Path("project_tourismpred/deployment/requirements.txt")
STREAMLIT_APP_URL = os.getenv("STREAMLIT_APP_URL", "").strip()

# One sample row with exactly the columns app.py sends to the model
SAMPLE_INPUT = pd.DataFrame([{
    "Age": 30, "TypeofContact": "Self Enquiry", "CityTier": 1, "Occupation": "Salaried",
    "Gender": "Male", "NumberOfPersonVisiting": 2, "PreferredPropertyStar": 3,
    "MaritalStatus": "Married", "NumberOfTrips": 2, "Passport": 1, "OwnCar": 1,
    "NumberOfChildrenVisiting": 0, "Designation": "Manager", "MonthlyIncome": 20000.0,
    "PitchSatisfactionScore": 3, "ProductPitched": "Basic", "NumberOfFollowups": 3,
    "DurationOfPitch": 15.0,
}])


def verify_deployment_artifacts():
    required_files = [MODEL_PATH, APP_PATH, REQ_PATH]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        print(f"ERROR: Missing deployment artifacts: {missing}")
        sys.exit(1)
    print(f"Verified deployment artifacts: {[str(p) for p in required_files]}")


# ✅ FIX (new): prove the committed model loads and predicts with the app's own requirements.
#    This is the check that actually tells you whether the Streamlit app will work.
def smoke_test_model():
    try:
        model = joblib.load(MODEL_PATH)
        pred = model.predict(SAMPLE_INPUT)[0]
        proba = model.predict_proba(SAMPLE_INPUT)[0, 1]
    except Exception as exc:
        print(f"ERROR: Model failed to load/predict: {exc}")
        sys.exit(1)
    print(f"Model smoke test passed: prediction={int(pred)}, P(purchase)={proba:.3f}")


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
    except Exception as exc:
        # ❌ ERROR (Deploy job): this used to call sys.exit(1). Streamlit Community Cloud
        #    answers bots with redirects/403s and sleeping apps can time out, so the job
        #    failed even when the app was fine. The app is redeployed by Streamlit Cloud
        #    itself, not by this job.
        # ✅ FIX: report it as a warning instead of failing the pipeline.
        print(f"::warning::Live app health check could not confirm the app: {exc}")


if __name__ == "__main__":
    verify_deployment_artifacts()
    smoke_test_model()
    trigger_and_verify_endpoint()
    print("End-to-end deployment stage completed successfully.")
