"""
FastAPI entrypoint for the ER triage system.

Run locally (from repo root):
    uvicorn api.main:app --reload

POST /triage runs the trained XGBoost model + SHAP explainer and returns
the predicted ESI level, confidence, and the top 3 SHAP-derived risk factors.
"""
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import TriageRequest, TriageResponse


# ─── Model artifact loading ─────────────────────────────────────────────
# Loaded once at import time; reused for every request.

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_VERSION = "xgb-1.0.0"

with open(MODELS_DIR / "triage_model.pkl", "rb") as f:
    MODEL = pickle.load(f)

with open(MODELS_DIR / "feature_names.pkl", "rb") as f:
    FEATURE_NAMES: list[str] = pickle.load(f)

with open(MODELS_DIR / "shap_explainer.pkl", "rb") as f:
    EXPLAINER = pickle.load(f)


# ─── Feature defaults (used only if the request omits a field) ──────────
# Clinical normal-ish values. Better than imputing with NaN since XGBoost
# accepts NaN but SHAP labels become misleading.
_FEATURE_DEFAULTS = {
    "heart_rate":        80.0,
    "systolic_bp":      120.0,
    "diastolic_bp":      80.0,
    "resp_rate":         16.0,
    "spo2":              98.0,
    "temperature":       98.6,   # Fahrenheit
}

# Plain-English labels for SHAP output (matches train_model.py).
_FEATURE_LABELS = {
    "heart_rate":        "Heart rate",
    "systolic_bp":       "Systolic blood pressure",
    "diastolic_bp":      "Diastolic blood pressure",
    "resp_rate":         "Respiratory rate",
    "spo2":              "Oxygen saturation (SpO2)",
    "temperature":       "Body temperature",
    "age":               "Patient age",
    "gender_m":          "Patient gender",
    "shock_index":       "Shock index (HR/SBP)",
    "has_diabetes":      "Diabetes history",
    "has_hypertension":  "Hypertension history",
    "has_heart_disease": "Heart disease history",
    "has_sepsis":        "Sepsis history",
    "has_resp_failure":  "Respiratory failure history",
    "complaint_cat":     "Chief complaint type",
}


def _request_to_features(req: TriageRequest) -> dict[str, float]:
    """Convert TriageRequest -> flat dict of feature values matching FEATURE_NAMES."""
    v = req.vitals
    c = req.comorbidities

    heart_rate    = v.heart_rate    if v.heart_rate    is not None else _FEATURE_DEFAULTS["heart_rate"]
    systolic_bp   = v.systolic_bp   if v.systolic_bp   is not None else _FEATURE_DEFAULTS["systolic_bp"]
    diastolic_bp  = v.diastolic_bp  if v.diastolic_bp  is not None else _FEATURE_DEFAULTS["diastolic_bp"]
    resp_rate     = v.resp_rate     if v.resp_rate     is not None else _FEATURE_DEFAULTS["resp_rate"]
    spo2          = v.spo2          if v.spo2          is not None else _FEATURE_DEFAULTS["spo2"]
    temperature   = v.temperature   if v.temperature   is not None else _FEATURE_DEFAULTS["temperature"]

    # shock_index = HR / SBP, guarded against zero
    shock_index = float(heart_rate) / float(systolic_bp) if systolic_bp else np.nan

    return {
        "heart_rate":        heart_rate,
        "systolic_bp":       systolic_bp,
        "diastolic_bp":      diastolic_bp,
        "resp_rate":         resp_rate,
        "spo2":              spo2,
        "temperature":       temperature,
        "age":               req.age,
        "gender_m":          1 if req.gender == "M" else 0,
        "has_diabetes":      c.has_diabetes,
        "has_hypertension":  c.has_hypertension,
        "has_heart_disease": c.has_heart_disease,
        "has_sepsis":        c.has_sepsis,
        "has_resp_failure":  c.has_resp_failure,
        "complaint_cat":     req.complaint_cat,
        "shock_index":       shock_index,
    }


def _format_top_factors(features: dict, shap_per_feature: np.ndarray) -> list[str]:
    """Pick the 3 features with the largest |SHAP value| and format them as strings."""
    contributions = list(zip(FEATURE_NAMES, shap_per_feature))
    contributions.sort(key=lambda kv: abs(kv[1]), reverse=True)

    out: list[str] = []
    for name, shap_val in contributions[:3]:
        label = _FEATURE_LABELS.get(name, name)
        value = features.get(name, "N/A")
        direction = "raises severity" if shap_val > 0 else "lowers severity"
        out.append(f"{label}={value} ({direction}, SHAP {shap_val:+.3f})")
    return out


# ─── App + endpoints ────────────────────────────────────────────────────

app = FastAPI(
    title="ER Triage System",
    description="Predicts ESI (Emergency Severity Index 1-5) at patient arrival.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "n_features": len(FEATURE_NAMES),
    }


@app.post("/triage", response_model=TriageResponse)
def triage(req: TriageRequest) -> TriageResponse:
    try:
        features = _request_to_features(req)

        # Build a 1-row DataFrame in the exact column order the model expects.
        X = pd.DataFrame([features])[FEATURE_NAMES]

        # Predict.
        proba = MODEL.predict_proba(X)[0]
        predicted_class = int(np.argmax(proba))      # 0..4
        esi_level = predicted_class + 1              # ESI 1..5
        confidence = float(proba[predicted_class])

        # SHAP for the predicted class.
        shap_obj = EXPLAINER(X)
        if hasattr(shap_obj, "values"):
            # Modern SHAP Explanation: shape (samples, features, classes)
            vals = shap_obj.values
            if vals.ndim == 3:
                shap_per_feature = vals[0][:, predicted_class]
            else:
                shap_per_feature = vals[0]
        else:
            # Legacy list-of-arrays return shape: list[(samples, features)] per class
            shap_per_feature = shap_obj[predicted_class][0]

        top_risk_factors = _format_top_factors(features, shap_per_feature)

        return TriageResponse(
            esi_level=esi_level,
            confidence=round(confidence, 4),
            top_risk_factors=top_risk_factors,
            model_version=MODEL_VERSION,
        )

    except Exception as e:
        # Don't leak internals to the client, but surface the type.
        raise HTTPException(status_code=500, detail=f"prediction failed: {type(e).__name__}: {e}")
