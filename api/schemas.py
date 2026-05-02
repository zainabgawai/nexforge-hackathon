"""Request / response schemas for the triage API."""
from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


# ─── Vitals ─────────────────────────────────────────────────────────────
class Vitals(BaseModel):
    heart_rate:   Optional[float] = Field(None, ge=0,  le=300, description="bpm")
    systolic_bp:  Optional[float] = Field(None, ge=0,  le=300, description="mmHg")
    diastolic_bp: Optional[float] = Field(None, ge=0,  le=200, description="mmHg")
    resp_rate:    Optional[float] = Field(None, ge=0,  le=80,  description="breaths/min")
    spo2:         Optional[float] = Field(None, ge=0,  le=100, description="%")
    temperature:  Optional[float] = Field(None, ge=80, le=115, description="Fahrenheit")

# ─── Comorbidities ──────────────────────────────────────────────────────
class Comorbidities(BaseModel):
    has_diabetes:      int = Field(0, ge=0, le=1)
    has_hypertension:  int = Field(0, ge=0, le=1)
    has_heart_disease: int = Field(0, ge=0, le=1)

# ─── Triage request ─────────────────────────────────────────────────────
class TriageRequest(BaseModel):
    age:            float         = Field(..., ge=0, le=120)
    gender:         Literal["M", "F"]
    vitals:         Vitals
    comorbidities:  Comorbidities = Field(default_factory=Comorbidities)
    complaint_cat:  int           = Field(
        0, ge=0, le=5,
        description="0=other, 1=cardiac, 2=infection, 3=respiratory, 4=neuro, 5=abdominal"
    )
    # Free-text chief complaint — encoded via sentence-transformer → PCA(8 dims)
    # and passed to the Gemini LLM for clinical reasoning.
    # Falls back to complaint_cat keyword encoding when omitted.
    complaint_text: str = Field(
        "",
        description="Free-text chief complaint (e.g. 'chest pain radiating to left arm, onset 2h ago')"
    )

class ClinicalSignal(BaseModel):
    flag:          str = Field(..., description="Short label, e.g. 'SEVERE TACHYCARDIA'")
    value:         str = Field(..., description="Measured value string, e.g. 'HR 155 bpm'")
    severity:      str = Field(..., description="'critical' | 'high' | 'moderate'")
    clinical_note: str = Field("",  description="One-line clinical interpretation")

class TriageResponse(BaseModel):
    model_config = {"protected_namespaces": ()}  

    # ── Core output ──────────────────────────────────────────
    esi_level:  int   = Field(..., ge=1, le=5,
                              description="Final ESI level (1=resuscitation … 5=non-urgent). "
                                          "May differ from model_esi when a clinical override fires.")
    model_esi:  int   = Field(..., ge=1, le=5,
                              description="Raw XGBoost prediction before any clinical override.")
    confidence: float = Field(..., ge=0, le=1,
                              description="Calibrated probability of the assigned class. "
                                          "Fixed at 0.95 when a clinical override is applied.")

    # ── Clinical override layer ───────────────────────────────
    override_applied: bool          = Field(...,
        description="True when apply_clinical_override() changed the model's ESI.")
    override_reason:  Optional[str] = Field(None,
        description="Human-readable reason for the override, or null.")

    # ── Human-review / uncertainty flag ──────────────────────
    needs_human_review: bool          = Field(...,
        description="True when confidence < 50% OR model and clinical signals disagree. "
                    "Treat as a mandatory escalation trigger.")
    uncertainty_reason: Optional[str] = Field(None,
        description="Explanation of why human review was triggered, or null.")

    # ── LLM output (Gemini or clinical fallback) ─────────────
    clinical_summary:       str       = Field("",
        description="One-sentence risk profile for the charge nurse.")
    top_clinical_findings:  list[str] = Field(default_factory=list,
        description="Top 3 clinically dangerous findings — based on vitals/signals, NOT SHAP scores.")
    immediate_next_steps:   list[str] = Field(default_factory=list,
        description="3–5 ordered clinical actions appropriate for this ESI level.")
    confidence_explanation: str       = Field("",
        description="One sentence explaining why this ESI was assigned or overridden.")

    # ── Clinical signal engine output ─────────────────────────
    clinical_signals: list[ClinicalSignal] = Field(default_factory=list,
        description="All flagged clinical findings from the rule-based signal engine.")
    critical_flags:   list[str]            = Field(default_factory=list,
        description="Most dangerous findings as plain strings (subset of clinical_signals).")
    context_flags:    list[str]            = Field(default_factory=list,
        description="Comorbidity / demographic risk factors.")

    # ── Full probability breakdown ────────────────────────────
    all_probabilities: dict[str, float] = Field(default_factory=dict,
        description="Calibrated probability per ESI class, e.g. {'ESI-1': 0.03, 'ESI-2': 0.71, ...}")

    model_version: str


# ─── Queue ──────────────────────────────────────────────────────────────

class QueueEntry(BaseModel):
    patient_id:            str
    arrival_time:          datetime
    waiting_minutes:       int             = Field(..., ge=0)
    esi_level:             int             = Field(..., ge=1, le=5)
    model_esi:             int             = Field(..., ge=1, le=5)
    override_applied:      bool            = False
    needs_human_review:    bool            = False
    confidence:            float           = Field(..., ge=0, le=1)
    age:                   float
    gender:                Literal["M", "F"]
    complaint_cat:         int             = Field(..., ge=0, le=5)
    complaint_label:       str
    complaint_text:        str             = ""
    top_clinical_findings: list[str]       = Field(default_factory=list)
    critical_flags:        list[str]       = Field(default_factory=list)


class QueueResponse(BaseModel):
    count:        int
    last_updated: datetime
    patients:     list[QueueEntry]


# ─── Beds ───────────────────────────────────────────────────────────────

BedStatusLiteral = Literal["available", "occupied", "cleaning"]


class BedStatus(BaseModel):
    bed_id:     str
    category:   str                = Field(..., description="resuscitation, trauma, monitored, general, fast_track")
    status:     BedStatusLiteral
    patient_id: Optional[str]      = None
    since:      Optional[datetime] = Field(None, description="When this status started")


class BedCategorySummary(BaseModel):
    total:     int
    available: int
    occupied:  int
    cleaning:  int


class BedsSummary(BaseModel):
    total:       int
    available:   int
    occupied:    int
    cleaning:    int
    by_category: dict[str, BedCategorySummary]


class BedsResponse(BaseModel):
    last_updated: datetime
    summary:      BedsSummary
    beds:         list[BedStatus]
