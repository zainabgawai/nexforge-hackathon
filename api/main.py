"""
FastAPI entrypoint for the ER triage system.

Run locally (from repo root):
    uvicorn api.main:app --reload

Endpoints:
    GET  /health   – liveness + model version
    POST /triage   – run model on patient data → ESI + clinical signals + LLM next steps
    GET  /queue    – patients sorted by severity (ESI asc, then arrival asc)
    GET  /beds     – mock bed allocation status across categories

Changes from original main.py (aligned with train_model_v5 + pipeline_v3):
  - Removed SHAP: model v5 no longer saves shap_explainer.pkl.
    Clinical explanations come from clinical_signal_engine (rule-based) + Gemini LLM.
  - Removed shock_index feature: dropped in pipeline_v3 (redundant — HR + SBP already present).
  - Removed has_sepsis / has_resp_failure: dropped in pipeline_v3 (post-diagnosis leakage).
  - Added complaint_text → sentence-embedding → PCA(8 dims) feature encoding at inference time.
    Falls back gracefully if sentence-transformers is not installed.
  - _request_to_features() now calls explain_prediction() from the training module
    logic re-implemented here directly so main.py stays self-contained.
  - TriageResponse now carries the full explain_prediction() output: override flags,
    human-review flag, clinical signals, LLM findings, and per-class probabilities.
  - QueueEntry carries override_applied, needs_human_review, critical_flags for
    dashboard escalation display.
"""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pickle
import uuid
import json
import os

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import (
    BedCategorySummary,
    BedsResponse,
    BedsSummary,
    BedStatus,
    ClinicalSignal,
    QueueEntry,
    QueueResponse,
    TriageRequest,
    TriageResponse,
)


# ─── Model artifact loading ─────────────────────────────────────────────

MODELS_DIR    = Path(__file__).resolve().parent.parent / "models"
MODEL_VERSION = "xgb-binary-1.0.0"

# ── BINARY MODEL (train_model_v6) ──
# The classifier predicts {0=non_critical (ESI 3-5), 1=critical (ESI 1-2)}.
# Granular ESI 1-5 is produced downstream by _apply_clinical_override().
with open(MODELS_DIR / "triage_model_binary.pkl", "rb") as f:
    MODEL = pickle.load(f)                          # CalibratedClassifierCV (binary)

with open(MODELS_DIR / "feature_names_binary.pkl", "rb") as f:
    FEATURE_NAMES: list[str] = pickle.load(f)       # ordered list used at training time

with open(MODELS_DIR / "real_median_binary.pkl", "rb") as f:
    _real_median_dict: dict = pickle.load(f)        # for NaN imputation at inference
REAL_MEDIAN = pd.Series(_real_median_dict)

# ── Sentence-embedding model (optional — graceful fallback) ──────────────
# Loaded once at startup; used to encode complaint_text → complaint_emb_* features.
_EMBED_MODEL   = None
_EMBED_PCA     = None
_EMB_COLS      = [f for f in FEATURE_NAMES if f.startswith("complaint_emb_")]

try:
    from sentence_transformers import SentenceTransformer
    with open(MODELS_DIR / "complaint_embed_model.pkl", "rb") as f:
        _saved = pickle.load(f)
    _EMBED_MODEL = SentenceTransformer(_saved["embed_model_name"])
    _EMBED_PCA   = _saved["pca"]
    print(f"[startup] Sentence-embedding model loaded ({len(_EMB_COLS)} PCA dims).")
except Exception as e:
    print(f"[startup] Sentence-embedding not available ({e}). "
          "complaint_emb_* features will use median imputation.")


# ─── Clinical override thresholds (mirror train_model_v5 CONFIG) ────────

_ESI1_VITALS = {
    "sbp_low":  80,
    "spo2_low": 88,
    "hr_high":  150,
    "hr_low":   40,
    "rr_high":  35,
    "temp_low": 95,
}
_ESI2_RULE = {
    "min_flags": 2,
    "sbp":       90,
    "spo2":      92,
    "hr":        130,
    "rr":        28,
    "shock_index": 1.0,
}
CONFIDENCE_THRESHOLD = 0.50


# ─── Feature defaults ───────────────────────────────────────────────────

_VITAL_DEFAULTS = {
    "heart_rate":   80.0,
    "systolic_bp":  120.0,
    "diastolic_bp": 80.0,
    "resp_rate":    16.0,
    "spo2":         98.0,
    "temperature":  98.6,
}


# ─── Complaint category labels ──────────────────────────────────────────

_COMPLAINT_LABELS = {
    0: "other",
    1: "cardiac",
    2: "infection",
    3: "respiratory",
    4: "neuro",
    5: "abdominal",
}


# ─── Feature builder ────────────────────────────────────────────────────

def _request_to_features(req: TriageRequest) -> dict:
    """
    Convert TriageRequest → flat feature dict matching FEATURE_NAMES.

    Feature set (from train_model_v5 FEATURE_COLS):
        heart_rate, systolic_bp, diastolic_bp, resp_rate, spo2, temperature,
        age, gender_m, complaint_cat,
        has_diabetes, has_hypertension, has_heart_disease,
        complaint_emb_0 … complaint_emb_7   (if embedding model available)

    Removed vs old main.py:
        shock_index      — dropped in pipeline_v3 (redundant)
        has_sepsis       — dropped in pipeline_v3 (post-diagnosis leakage)
        has_resp_failure — dropped in pipeline_v3 (post-diagnosis leakage)
    """
    v = req.vitals
    c = req.comorbidities

    hr   = v.heart_rate   if v.heart_rate   is not None else _VITAL_DEFAULTS["heart_rate"]
    sbp  = v.systolic_bp  if v.systolic_bp  is not None else _VITAL_DEFAULTS["systolic_bp"]
    dbp  = v.diastolic_bp if v.diastolic_bp is not None else _VITAL_DEFAULTS["diastolic_bp"]
    rr   = v.resp_rate    if v.resp_rate    is not None else _VITAL_DEFAULTS["resp_rate"]
    spo2 = v.spo2         if v.spo2         is not None else _VITAL_DEFAULTS["spo2"]
    tmp  = v.temperature  if v.temperature  is not None else _VITAL_DEFAULTS["temperature"]

    features: dict = {
        "heart_rate":        hr,
        "systolic_bp":       sbp,
        "diastolic_bp":      dbp,
        "resp_rate":         rr,
        "spo2":              spo2,
        "temperature":       tmp,
        "age":               req.age,
        "gender_m":          1 if req.gender == "M" else 0,
        "complaint_cat":     req.complaint_cat,
        "has_diabetes":      c.has_diabetes,
        "has_hypertension":  c.has_hypertension,
        "has_heart_disease": c.has_heart_disease,
    }

    # ── Complaint text → sentence embedding → PCA ─────────────────────
    if _EMB_COLS:
        if req.complaint_text and _EMBED_MODEL and _EMBED_PCA:
            try:
                raw_emb = _EMBED_MODEL.encode([req.complaint_text])
                reduced = _EMBED_PCA.transform(raw_emb)[0]
                for i, col in enumerate(_EMB_COLS):
                    features[col] = float(reduced[i])
            except Exception:
                # Fall through to median imputation below
                for col in _EMB_COLS:
                    features[col] = float(REAL_MEDIAN.get(col, 0.0))
        else:
            # No complaint text or embedding model unavailable — use training median
            for col in _EMB_COLS:
                features[col] = float(REAL_MEDIAN.get(col, 0.0))

    return features


# ─── Clinical override (mirrors apply_clinical_override in train_model_v5) ──

def _apply_clinical_override(model_esi: int,
                              features: dict) -> tuple[int, str | None]:
    hr   = features.get("heart_rate",   0)
    sbp  = features.get("systolic_bp",  999)
    spo2 = features.get("spo2",         100)
    rr   = features.get("resp_rate",    0)
    tmp  = features.get("temperature",  98.6)

    # ESI-1: immediate life threat
    esi1 = []
    if sbp  < _ESI1_VITALS["sbp_low"]:  esi1.append(f"critical hypotension (SBP {sbp} mmHg)")
    if spo2 < _ESI1_VITALS["spo2_low"]: esi1.append(f"severe hypoxia (SpO₂ {spo2}%)")
    if hr   > _ESI1_VITALS["hr_high"]:  esi1.append(f"severe tachycardia (HR {hr} bpm)")
    if hr   < _ESI1_VITALS["hr_low"]:   esi1.append(f"severe bradycardia (HR {hr} bpm)")
    if rr   > _ESI1_VITALS["rr_high"]:  esi1.append(f"respiratory failure risk (RR {rr}/min)")
    if tmp  < _ESI1_VITALS["temp_low"]: esi1.append(f"severe hypothermia (Temp {tmp}°F)")
    if esi1:
        return 1, "ESI-1 OVERRIDE: " + "; ".join(esi1)

    # Downward review: model says ESI-1 but vitals are only borderline
    if model_esi == 1:
        borderline = []
        if 80 <= sbp  < 90:  borderline.append(f"SBP borderline ({sbp} mmHg)")
        if 88 <= spo2 < 92:  borderline.append(f"SpO₂ borderline ({spo2}%)")
        if 130 < hr  <= 150: borderline.append(f"HR borderline ({hr} bpm)")
        if 28  < rr  <= 35:  borderline.append(f"RR borderline ({rr}/min)")
        if borderline:
            return 1, ("ESI-1 REVIEW FLAG: model predicted ESI-1 but vitals are borderline — "
                       + "; ".join(borderline) + ". Human verification recommended.")

    # ESI-2: high-risk situation — upgrade only if model is lower
    if model_esi >= 3:
        esi2 = []
        si = hr / sbp if sbp > 0 else 0
        if sbp  < _ESI2_RULE["sbp"]:          esi2.append(f"hypotension (SBP {sbp} mmHg)")
        if spo2 < _ESI2_RULE["spo2"]:         esi2.append(f"hypoxia (SpO₂ {spo2}%)")
        if hr   > _ESI2_RULE["hr"]:            esi2.append(f"tachycardia (HR {hr} bpm)")
        if rr   > _ESI2_RULE["rr"]:            esi2.append(f"tachypnea (RR {rr}/min)")
        if si   > _ESI2_RULE["shock_index"]:   esi2.append(f"elevated shock index ({si:.2f})")
        if len(esi2) >= _ESI2_RULE["min_flags"]:
            return 2, "ESI-2 OVERRIDE: concurrent high-risk vitals — " + "; ".join(esi2)

    return model_esi, None


# ─── Clinical signal engine (mirrors clinical_signal_engine in train_model_v5) ─

def _clinical_signal_engine(features: dict) -> dict:
    hr   = features.get("heart_rate")
    sbp  = features.get("systolic_bp")
    rr   = features.get("resp_rate")
    spo2 = features.get("spo2")
    tmp  = features.get("temperature")
    age  = features.get("age", 0)

    signals:  list[dict] = []
    critical: list[str]  = []
    context:  list[str]  = []

    # Cardiovascular
    if hr is not None:
        if hr > 150:
            signals.append({"flag": "SEVERE TACHYCARDIA", "value": f"HR {hr} bpm",
                            "severity": "critical",
                            "clinical_note": "HR >150 suggests SVT, VT, or decompensated shock"})
            critical.append(f"Severe tachycardia (HR {hr} bpm)")
        elif hr > 100:
            signals.append({"flag": "TACHYCARDIA", "value": f"HR {hr} bpm",
                            "severity": "high",
                            "clinical_note": "Tachycardia may indicate pain, fever, hypovolemia, or arrhythmia"})
            critical.append(f"Tachycardia (HR {hr} bpm)")
        elif hr < 40:
            signals.append({"flag": "SEVERE BRADYCARDIA", "value": f"HR {hr} bpm",
                            "severity": "critical",
                            "clinical_note": "HR <40 — risk of cardiac output failure"})
            critical.append(f"Severe bradycardia (HR {hr} bpm)")
        elif hr < 60:
            signals.append({"flag": "BRADYCARDIA", "value": f"HR {hr} bpm",
                            "severity": "moderate",
                            "clinical_note": "May be medication-related or cardiac conduction issue"})

    if sbp is not None:
        if sbp < 80:
            signals.append({"flag": "CRITICAL HYPOTENSION", "value": f"SBP {sbp} mmHg",
                            "severity": "critical",
                            "clinical_note": "Cardiogenic, distributive, or haemorrhagic shock"})
            critical.append(f"Critical hypotension (SBP {sbp} mmHg)")
        elif sbp < 90:
            signals.append({"flag": "HYPOTENSION", "value": f"SBP {sbp} mmHg",
                            "severity": "high",
                            "clinical_note": "Borderline perfusion pressure — monitor closely"})
            critical.append(f"Hypotension (SBP {sbp} mmHg)")
        elif sbp > 180:
            signals.append({"flag": "HYPERTENSIVE URGENCY", "value": f"SBP {sbp} mmHg",
                            "severity": "high",
                            "clinical_note": "Assess for end-organ damage"})

    if hr and sbp and sbp > 0:
        si = hr / sbp
        if si > 1.0:
            signals.append({"flag": "ELEVATED SHOCK INDEX", "value": f"SI {si:.2f}",
                            "severity": "high",
                            "clinical_note": "Shock index >1.0 correlates with haemodynamic instability"})
            critical.append(f"Shock index {si:.2f} (normal <0.7)")

    # Respiratory
    if spo2 is not None:
        if spo2 < 88:
            signals.append({"flag": "SEVERE HYPOXIA", "value": f"SpO₂ {spo2}%",
                            "severity": "critical",
                            "clinical_note": "SpO₂ <88% — immediate O₂ therapy and airway assessment"})
            critical.append(f"Severe hypoxia (SpO₂ {spo2}%)")
        elif spo2 < 92:
            signals.append({"flag": "HYPOXIA", "value": f"SpO₂ {spo2}%",
                            "severity": "high",
                            "clinical_note": "SpO₂ <92% — supplemental oxygen indicated"})
            critical.append(f"Hypoxia (SpO₂ {spo2}%)")
        elif spo2 < 95:
            signals.append({"flag": "LOW SpO₂", "value": f"SpO₂ {spo2}%",
                            "severity": "moderate",
                            "clinical_note": "Below normal — monitor and consider oxygen"})

    if rr is not None:
        if rr > 30:
            signals.append({"flag": "SEVERE TACHYPNEA", "value": f"RR {rr}/min",
                            "severity": "critical",
                            "clinical_note": "RR >30 — respiratory distress or failure"})
            critical.append(f"Severe tachypnea (RR {rr}/min)")
        elif rr > 20:
            signals.append({"flag": "TACHYPNEA", "value": f"RR {rr}/min",
                            "severity": "high",
                            "clinical_note": "Elevated RR — assess for infection, pain, or respiratory compromise"})

    # Temperature
    if tmp is not None:
        if tmp < 95:
            signals.append({"flag": "SEVERE HYPOTHERMIA", "value": f"Temp {tmp}°F",
                            "severity": "critical",
                            "clinical_note": "Risk of cardiac arrhythmia and coagulopathy"})
            critical.append(f"Severe hypothermia (Temp {tmp}°F)")
        elif tmp > 103:
            signals.append({"flag": "HIGH FEVER", "value": f"Temp {tmp}°F",
                            "severity": "high",
                            "clinical_note": "Fever >103°F — consider sepsis workup"})
        elif tmp > 100.4:
            signals.append({"flag": "FEVER", "value": f"Temp {tmp}°F",
                            "severity": "moderate",
                            "clinical_note": "Low-grade fever — monitor, consider infection"})

    # Context / comorbidities
    if age > 65:
        context.append(f"Elderly patient (age {int(age)}) — higher risk of atypical presentation")
    if features.get("has_diabetes"):
        context.append("Diabetes — risk of silent MI, DKA, hypoglycaemia")
    if features.get("has_hypertension"):
        context.append("Hypertension — assess for hypertensive emergency")
    if features.get("has_heart_disease"):
        context.append("Known heart disease — prioritise cardiac workup")

    return {
        "signals":        signals,
        "critical_flags": critical,
        "context_flags":  context,
        "signal_count":   len(signals),
        "critical_count": len(critical),
    }


# ─── Clinical fallback (no LLM) ─────────────────────────────────────────

_ESI_ACTIONS: dict[int, list[str]] = {
    1: [
        "Activate resuscitation team immediately — patient meets ESI-1 criteria",
        "Establish two large-bore IV lines and begin fluid resuscitation",
        "Continuous cardiac monitoring, pulse oximetry, and capnography",
        "Obtain 12-lead ECG, ABG, and STAT labs (BMP, CBC, lactate, troponin)",
        "Prepare for airway management — have RSI medications at bedside",
    ],
    2: [
        "Place in monitored bay within 10 minutes",
        "Obtain 12-lead ECG and IV access immediately",
        "STAT labs: BMP, CBC, troponin, lactate, coagulation panel",
        "Supplemental oxygen if SpO₂ <94%",
        "Notify attending physician immediately",
    ],
    3: [
        "Assign monitored room — 2+ resources required",
        "IV access, basic labs (BMP, CBC), and relevant imaging",
        "Vital sign reassessment every 30 minutes",
        "Pain or symptom management per protocol",
    ],
    4: [
        "Assign non-monitored room — 1 resource required",
        "Focused history and physical exam",
        "Targeted diagnostic test if indicated",
        "Routine vital sign monitoring",
    ],
    5: [
        "Fast-track or waiting-room assessment appropriate",
        "No resources required beyond prescription or wound care",
        "Vital signs once on arrival",
    ],
}


def _clinical_fallback(clinical_output: dict, esi_level: int,
                       override_reason: str | None) -> dict:
    findings = []
    for sig in clinical_output["signals"][:3]:
        findings.append(f"{sig['flag']} ({sig['value']}): {sig.get('clinical_note', '')}")
    if not findings:
        findings = ["No critical vital sign abnormalities detected at this time"]
    return {
        "top_3_clinical_findings": findings,
        "immediate_next_steps":    _ESI_ACTIONS.get(esi_level, _ESI_ACTIONS[3]),
        "clinical_summary": (
            f"ESI {esi_level} — "
            + (f"{len(clinical_output['critical_flags'])} critical vital sign abnormalities detected"
               if clinical_output["critical_flags"]
               else "vitals within acceptable range at this time")
        ),
        "confidence_explanation": override_reason or f"Patient vitals consistent with ESI {esi_level}",
    }


# ─── Full prediction (mirrors explain_prediction in train_model_v5) ──────

def _refine_non_critical_esi(features: dict) -> int:
    """
    For non-critical patients (model predicted ESI 3-5), assign a specific
    ESI based on simple rule-based heuristics so the queue shows variety.
    The model itself only decides critical/non-critical; this fans out to 3/4/5.
    """
    age   = features.get("age", 50)
    hr    = features.get("heart_rate",   80)
    sbp   = features.get("systolic_bp", 120)
    spo2  = features.get("spo2",         98)
    has_comorbidity = bool(
        features.get("has_diabetes", 0)
        or features.get("has_hypertension", 0)
        or features.get("has_heart_disease", 0)
    )
    vitals_borderline = (
        hr > 100 or hr < 60
        or sbp < 110 or sbp > 160
        or spo2 < 95
    )
    if vitals_borderline or (age > 65 and has_comorbidity):
        return 3   # urgent — needs monitoring
    if age < 40 and not has_comorbidity:
        return 5   # non-urgent — fast track candidate
    return 4       # less urgent — middle ground


def _predict(req: TriageRequest) -> dict:
    features = _request_to_features(req)

    # Build DataFrame in exact column order the model expects
    X = pd.DataFrame([features])[FEATURE_NAMES].fillna(REAL_MEDIAN).astype(np.float32)

    # ── Binary model prediction ──
    # proba shape: [p_non_critical, p_critical]
    proba          = MODEL.predict_proba(X)[0]
    non_crit_prob  = float(proba[0])
    critical_prob  = float(proba[1])
    model_critical = critical_prob >= 0.5

    # Map binary decision → baseline ESI 1-5 for the override layer.
    # Critical → ESI 2 baseline; override may force ESI 1 on severe vitals.
    # Non-critical → fan out to 3/4/5 based on age + vitals + comorbidities.
    if model_critical:
        model_esi = 2
        raw_conf  = critical_prob
    else:
        model_esi = _refine_non_critical_esi(features)
        raw_conf  = non_crit_prob

    # Clinical signal engine (independent of model)
    clinical_output = _clinical_signal_engine(features)

    # Clinical override layer
    final_esi, override_reason = _apply_clinical_override(model_esi, features)

    # Uncertainty / human-review trigger
    needs_human_review = False
    uncertainty_reason = None
    if raw_conf < CONFIDENCE_THRESHOLD:
        needs_human_review = True
        uncertainty_reason = (f"Low model confidence ({raw_conf:.1%}) — "
                              "prediction is uncertain. Human review required.")
    elif clinical_output["critical_count"] > 0 and final_esi > 2:
        needs_human_review = True
        uncertainty_reason = (f"Model/clinical signal disagreement: model says ESI-{final_esi} "
                              f"but {clinical_output['critical_count']} critical vital(s) flagged. "
                              "Human review required.")
    elif clinical_output["critical_count"] == 0 and final_esi <= 2 and not override_reason:
        needs_human_review = True
        uncertainty_reason = (f"Model predicts ESI-{final_esi} but no critical vitals flagged. "
                              "Verify clinical picture before acting.")

    # Confidence display
    if override_reason:
        confidence = 0.95
    elif clinical_output["critical_count"] > 0 and final_esi <= 2:
        confidence = max(raw_conf, 0.80)
    else:
        confidence = raw_conf

    # LLM (Gemini) or clinical fallback
    llm_result = _call_gemini(features, clinical_output, final_esi,
                               override_reason, req.complaint_text)

    return {
        "esi_level":               final_esi,
        "model_esi":               model_esi,
        "override_applied":        override_reason is not None,
        "override_reason":         override_reason,
        "needs_human_review":      needs_human_review,
        "uncertainty_reason":      uncertainty_reason,
        "confidence":              round(confidence, 4),
        "clinical_summary":        llm_result.get("clinical_summary", ""),
        "top_clinical_findings":   llm_result.get("top_3_clinical_findings", []),
        "immediate_next_steps":    llm_result.get("immediate_next_steps", []),
        "confidence_explanation":  llm_result.get("confidence_explanation", ""),
        "clinical_signals":        clinical_output["signals"],
        "critical_flags":          clinical_output["critical_flags"],
        "context_flags":           clinical_output["context_flags"],
        # Truth from the binary model
        "model_critical":          model_critical,
        "critical_probability":    round(critical_prob, 4),
        "binary_probabilities": {
            "critical":     round(critical_prob, 4),
            "non_critical": round(non_crit_prob, 4),
        },
        # 5-class shape for frontend backward compatibility — synthesized,
        # NOT from the model. Mass concentrates on {ESI-2, ESI-3} mirroring
        # the binary split; final ESI is decided by the override layer above.
        "all_probabilities": {
            "ESI-1": round(0.15 * critical_prob, 4),
            "ESI-2": round(0.85 * critical_prob, 4),
            "ESI-3": round(0.70 * non_crit_prob, 4),
            "ESI-4": round(0.20 * non_crit_prob, 4),
            "ESI-5": round(0.10 * non_crit_prob, 4),
        },
    }


# ─── Gemini call ────────────────────────────────────────────────────────

def _call_gemini(features: dict, clinical_output: dict,
                 esi_level: int, override_reason: str | None,
                 complaint_text: str) -> dict:
    try:
        from google import genai
        _gemini = genai.Client()

        critical_str = "\n".join(f"  • {f}" for f in clinical_output["critical_flags"]) \
                       or "  • None detected"
        context_str  = "\n".join(f"  • {f}" for f in clinical_output["context_flags"]) \
                       or "  • None"
        vitals_str   = (
            f"HR={features.get('heart_rate')} bpm, "
            f"BP={features.get('systolic_bp')}/{features.get('diastolic_bp')} mmHg, "
            f"SpO₂={features.get('spo2')}%, "
            f"RR={features.get('resp_rate')}/min, "
            f"Temp={features.get('temperature')}°F, "
            f"Age={features.get('age')}"
        )
        override_str = f"\n⚠️ CLINICAL OVERRIDE APPLIED: {override_reason}" if override_reason else ""

        prompt = f"""You are a board-certified emergency medicine physician writing a triage note.

Patient vitals: {vitals_str}
Chief complaint: "{complaint_text or 'Not provided'}"
Assigned ESI Level: {esi_level} (1=immediate, 2=emergent, 3=urgent, 4=less urgent, 5=non-urgent){override_str}

Critical clinical findings:
{critical_str}

Contextual risk factors:
{context_str}

Instructions:
- top_3_clinical_findings: List the 3 most clinically dangerous findings for THIS patient.
  Base these ONLY on the vitals and clinical findings above — not on statistical model scores.
  Each entry must name the finding AND explain its specific danger for this patient.
- immediate_next_steps: List 3-5 SPECIFIC, ordered clinical actions appropriate for ESI {esi_level}.
  Be concrete (e.g. "Obtain 12-lead ECG immediately — exclude STEMI given chest pain + tachycardia").
  Actions must be ordered by urgency.
- clinical_summary: One sentence risk profile for the charge nurse.
- confidence_explanation: One sentence explaining why this ESI level was assigned (or overridden).

Respond ONLY with valid JSON, no markdown, no extra text:
{{
  "top_3_clinical_findings": ["finding 1", "finding 2", "finding 3"],
  "immediate_next_steps": ["step 1", "step 2", "step 3"],
  "clinical_summary": "...",
  "confidence_explanation": "..."
}}"""

        response = _gemini.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt,
            config={"temperature": 0.2},
        )
        text = response.text if hasattr(response, "text") and response.text \
               else response.contents[0].text
        return json.loads(text)

    except Exception as e:
        print(f"  [Gemini unavailable: {e} — using clinical fallback]")
        return _clinical_fallback(clinical_output, esi_level, override_reason)


# ─── In-memory queue + beds ─────────────────────────────────────────────

_QUEUE: list[QueueEntry] = []
_BEDS:  list[BedStatus]  = []


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_patient_id() -> str:
    return f"P-{uuid.uuid4().hex[:6].upper()}"


def _seed_queue() -> None:
    if _QUEUE:
        return
    now = _now()
    samples = [
        # (arrived_min_ago, esi, model_esi, override, needs_review, conf, age, gender, cc, top3_findings, critical_flags)
        (22, 1, 3, True, False, 0.95, 78, "F", 4,
         ["Severe hypoxia (SpO₂ 88%): inadequate O₂ delivery, risk of cardiac arrest",
          "Severe tachycardia (HR 155 bpm): suggests decompensated shock or arrhythmia",
          "Respiratory failure risk (RR 36/min): immediate airway management required"],
         ["Severe hypoxia (SpO₂ 88%)", "Severe tachycardia (HR 155 bpm)"]),
        (12, 2, 2, False, False, 0.89, 65, "M", 1,
         ["Tachycardia (HR 128 bpm): possible ACS or haemodynamic compromise",
          "Hypertension history: assess for hypertensive emergency",
          "Systolic BP 162 mmHg: hypertensive urgency — end-organ damage risk"],
         ["Tachycardia (HR 128 bpm)"]),
        (45, 3, 3, False, True, 0.61, 42, "F", 5,
         ["Fever 101.4°F: consider sepsis workup",
          "Abdominal complaint — possible surgical abdomen",
          "Tachycardia (HR 104 bpm): may indicate pain or early infection"],
         ["Tachycardia (HR 104 bpm)"]),
        (8,  4, 4, False, False, 0.74, 31, "M", 0,
         ["Mild tachycardia (HR 92 bpm): likely pain-related",
          "No critical vital abnormalities detected",
          "Young healthy patient — lower baseline risk"],
         []),
        (3,  5, 5, False, False, 0.86, 24, "F", 0,
         ["Vitals entirely within normal limits",
          "No comorbidities reported",
          "Non-urgent complaint — appropriate for fast-track"],
         []),
    ]
    for (offset, esi, m_esi, override, review, conf, age, gender,
         cc, findings, crit) in samples:
        _QUEUE.append(QueueEntry(
            patient_id=_new_patient_id(),
            arrival_time=now - timedelta(minutes=offset),
            waiting_minutes=offset,
            esi_level=esi,
            model_esi=m_esi,
            override_applied=override,
            needs_human_review=review,
            confidence=conf,
            age=age,
            gender=gender,
            complaint_cat=cc,
            complaint_label=_COMPLAINT_LABELS[cc],
            top_clinical_findings=findings,
            critical_flags=crit,
        ))


def _seed_beds() -> None:
    if _BEDS:
        return
    now = _now()
    by_esi = {p.esi_level: p for p in _QUEUE}

    def occ(esi: int) -> tuple[str | None, datetime | None]:
        p = by_esi.get(esi)
        return (p.patient_id, p.arrival_time) if p else (None, None)

    pid1, t1 = occ(1); pid2, t2 = occ(2)
    pid3, t3 = occ(3); pid4, t4 = occ(4); pid5, t5 = occ(5)

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


# ─── App ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ER Triage System",
    description=(
        "Predicts ESI (Emergency Severity Index 1–5) at patient arrival using "
        "XGBoost + clinical override rules + Gemini LLM clinical reasoning."
    ),
    version="2.0.0",
)


@app.get("/health")
def health() -> dict:
    return {
        "status":        "ok",
        "model_version": MODEL_VERSION,
        "n_features":    len(FEATURE_NAMES),
        "embedding_model_loaded": _EMBED_MODEL is not None,
    }


@app.post("/triage", response_model=TriageResponse)
def triage(req: TriageRequest) -> TriageResponse:
    try:
        result = _predict(req)

        # Append to in-memory queue
        _QUEUE.append(QueueEntry(
            patient_id=_new_patient_id(),
            arrival_time=_now(),
            waiting_minutes=0,
            esi_level=result["esi_level"],
            model_esi=result["model_esi"],
            override_applied=result["override_applied"],
            needs_human_review=result["needs_human_review"],
            confidence=round(result["confidence"], 4),
            age=req.age,
            gender=req.gender,
            complaint_cat=req.complaint_cat,
            complaint_label=_COMPLAINT_LABELS.get(req.complaint_cat, "other"),
            complaint_text=req.complaint_text,
            top_clinical_findings=result["top_clinical_findings"],
            critical_flags=result["critical_flags"],
        ))

        return TriageResponse(
            esi_level=result["esi_level"],
            model_esi=result["model_esi"],
            model_critical=result["model_critical"],
            critical_probability=result["critical_probability"],
            confidence=round(result["confidence"], 4),
            override_applied=result["override_applied"],
            override_reason=result["override_reason"],
            needs_human_review=result["needs_human_review"],
            uncertainty_reason=result["uncertainty_reason"],
            clinical_summary=result["clinical_summary"],
            top_clinical_findings=result["top_clinical_findings"],
            immediate_next_steps=result["immediate_next_steps"],
            confidence_explanation=result["confidence_explanation"],
            clinical_signals=[ClinicalSignal(**s) for s in result["clinical_signals"]],
            critical_flags=result["critical_flags"],
            context_flags=result["context_flags"],
            binary_probabilities=result["binary_probabilities"],
            all_probabilities=result["all_probabilities"],
            model_version=MODEL_VERSION,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"prediction failed: {type(e).__name__}: {e}")


@app.get("/queue", response_model=QueueResponse)
def queue() -> QueueResponse:
    """All patients sorted by ESI (1=most urgent) then arrival time."""
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
    """Per-bed status and per-category + overall summary."""
    by_cat: dict[str, dict[str, int]] = {}
    avail = occ = clean = 0

    for b in _BEDS:
        cat = by_cat.setdefault(b.category,
                                {"total": 0, "available": 0, "occupied": 0, "cleaning": 0})
        cat["total"]  += 1
        cat[b.status] += 1
        if b.status == "available": avail += 1
        elif b.status == "occupied": occ   += 1
        elif b.status == "cleaning": clean += 1

    summary = BedsSummary(
        total=len(_BEDS), available=avail, occupied=occ, cleaning=clean,
        by_category={name: BedCategorySummary(**counts) for name, counts in by_cat.items()},
    )
    return BedsResponse(last_updated=_now(), summary=summary, beds=_BEDS)
