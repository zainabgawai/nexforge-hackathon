"""
FastAPI entrypoint for the ER triage system.

Run locally (from repo root):
    uvicorn api.main:app --reload

Endpoints:
    GET  /health   – liveness + model version
    POST /triage   – run model on patient data, return ESI + SHAP top-3
    GET  /queue    – patients sorted by severity (ESI asc, then arrival asc)
    GET  /beds     – mock bed allocation status across categories
"""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pickle
import uuid

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import (
    BedCategorySummary,
    BedsResponse,
    BedsSummary,
    BedStatus,
    QueueEntry,
    QueueResponse,
    TriageRequest,
    TriageResponse,
)


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


# ─── In-memory state for /queue and /beds ──────────────────────────────
# Process-local; resets when uvicorn restarts (--reload also resets it).
# Fine for the demo; would need a real store for multi-worker / production.

_COMPLAINT_LABELS = {
    0: "other",
    1: "cardiac",
    2: "infection",
    3: "respiratory",
    4: "neuro",
    5: "abdominal",
}

_QUEUE: list[QueueEntry] = []
_BEDS: list[BedStatus] = []


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_patient_id() -> str:
    return f"P-{uuid.uuid4().hex[:6].upper()}"


def _seed_queue() -> None:
    """A few demo patients spanning ESI 1-5 so /queue isn't empty on first hit."""
    if _QUEUE:
        return
    now = _now()
    samples = [
        # (arrived_min_ago, esi, confidence, age, gender, complaint_cat, top3)
        (22, 1, 0.97, 78, "F", 4,
         ["Shock index (HR/SBP)=1.55 (raises severity, SHAP +1.234)",
          "Oxygen saturation (SpO2)=88 (raises severity, SHAP +0.920)",
          "Respiratory rate=28 (raises severity, SHAP +0.710)"]),
        (12, 2, 0.89, 65, "M", 1,
         ["Heart rate=128 (raises severity, SHAP +0.830)",
          "Hypertension history=1 (raises severity, SHAP +0.410)",
          "Systolic blood pressure=162 (raises severity, SHAP +0.320)"]),
        (45, 3, 0.81, 42, "F", 5,
         ["Body temperature=101.4 (raises severity, SHAP +0.550)",
          "Patient age=42 (lowers severity, SHAP -0.210)",
          "Diabetes history=0 (lowers severity, SHAP -0.150)"]),
        (8, 4, 0.74, 31, "M", 0,
         ["Patient age=31 (lowers severity, SHAP -0.620)",
          "Heart rate=92 (lowers severity, SHAP -0.300)",
          "Oxygen saturation (SpO2)=98 (lowers severity, SHAP -0.180)"]),
        (3, 5, 0.86, 24, "F", 0,
         ["Patient age=24 (lowers severity, SHAP -0.780)",
          "Body temperature=98.4 (lowers severity, SHAP -0.420)",
          "Heart rate=76 (lowers severity, SHAP -0.250)"]),
    ]
    for offset_min, esi, conf, age, gender, cc, top3 in samples:
        _QUEUE.append(QueueEntry(
            patient_id=_new_patient_id(),
            arrival_time=now - timedelta(minutes=offset_min),
            waiting_minutes=offset_min,
            esi_level=esi,
            confidence=conf,
            age=age,
            gender=gender,
            complaint_cat=cc,
            complaint_label=_COMPLAINT_LABELS[cc],
            top_risk_factors=top3,
        ))


def _seed_beds() -> None:
    """Mock 13-bed ER. Some occupied (assigned to seeded queue patients), some free, one cleaning."""
    if _BEDS:
        return
    now = _now()

    # Map seeded patients by ESI for sensible bed assignment.
    by_esi = {p.esi_level: p for p in _QUEUE}

    def occ_for(esi: int) -> tuple[str | None, datetime | None]:
        p = by_esi.get(esi)
        return (p.patient_id, p.arrival_time) if p else (None, None)

    pid1, t1 = occ_for(1)
    pid2, t2 = occ_for(2)
    pid3, t3 = occ_for(3)
    pid4, t4 = occ_for(4)
    pid5, t5 = occ_for(5)

    _BEDS.extend([
        BedStatus(bed_id="RES-1",    category="resuscitation", status="occupied",  patient_id=pid1, since=t1),
        BedStatus(bed_id="RES-2",    category="resuscitation", status="available"),
        BedStatus(bed_id="TRAUMA-1", category="trauma",        status="cleaning",  since=now - timedelta(minutes=4)),
        BedStatus(bed_id="TRAUMA-2", category="trauma",        status="available"),
        BedStatus(bed_id="MON-1",    category="monitored",     status="occupied",  patient_id=pid2, since=t2),
        BedStatus(bed_id="MON-2",    category="monitored",     status="occupied",  patient_id=pid3, since=t3),
        BedStatus(bed_id="MON-3",    category="monitored",     status="available"),
        BedStatus(bed_id="MON-4",    category="monitored",     status="available"),
        BedStatus(bed_id="GEN-1",    category="general",       status="occupied",  patient_id=pid4, since=t4),
        BedStatus(bed_id="GEN-2",    category="general",       status="available"),
        BedStatus(bed_id="GEN-3",    category="general",       status="available"),
        BedStatus(bed_id="FT-1",     category="fast_track",    status="occupied",  patient_id=pid5, since=t5),
        BedStatus(bed_id="FT-2",     category="fast_track",    status="available"),
    ])


_seed_queue()
_seed_beds()


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

        # Append to the in-memory queue so /queue reflects this triage call.
        _QUEUE.append(QueueEntry(
            patient_id=_new_patient_id(),
            arrival_time=_now(),
            waiting_minutes=0,
            esi_level=esi_level,
            confidence=round(confidence, 4),
            age=req.age,
            gender=req.gender,
            complaint_cat=req.complaint_cat,
            complaint_label=_COMPLAINT_LABELS.get(req.complaint_cat, "other"),
            top_risk_factors=top_risk_factors,
        ))

        return TriageResponse(
            esi_level=esi_level,
            confidence=round(confidence, 4),
            top_risk_factors=top_risk_factors,
            model_version=MODEL_VERSION,
        )

    except HTTPException:
        raise
    except Exception as e:
        # Don't leak internals to the client, but surface the type.
        raise HTTPException(status_code=500, detail=f"prediction failed: {type(e).__name__}: {e}")


@app.get("/queue", response_model=QueueResponse)
def queue() -> QueueResponse:
    """All patients currently in the queue, sorted by ESI (1=most urgent) then arrival time."""
    now = _now()
    sorted_pts = sorted(_QUEUE, key=lambda p: (p.esi_level, p.arrival_time))
    refreshed = [
        p.model_copy(update={
            "waiting_minutes": max(0, int((now - p.arrival_time).total_seconds() // 60))
        })
        for p in sorted_pts
    ]
    return QueueResponse(count=len(refreshed), last_updated=now, patients=refreshed)


@app.get("/beds", response_model=BedsResponse)
def beds() -> BedsResponse:
    """Mock bed allocation: per-bed status + per-category and overall summary."""
    by_cat: dict[str, dict[str, int]] = {}
    avail = occ = clean = 0

    for b in _BEDS:
        cat = by_cat.setdefault(
            b.category,
            {"total": 0, "available": 0, "occupied": 0, "cleaning": 0},
        )
        cat["total"] += 1
        cat[b.status] += 1
        if b.status == "available":
            avail += 1
        elif b.status == "occupied":
            occ += 1
        elif b.status == "cleaning":
            clean += 1

    summary = BedsSummary(
        total=len(_BEDS),
        available=avail,
        occupied=occ,
        cleaning=clean,
        by_category={
            name: BedCategorySummary(**counts) for name, counts in by_cat.items()
        },
    )
    return BedsResponse(last_updated=_now(), summary=summary, beds=_BEDS)
