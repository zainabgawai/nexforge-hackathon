"""Request / response schemas for the triage API."""
from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


class Vitals(BaseModel):
    heart_rate:    Optional[float] = Field(None, ge=0, le=300, description="bpm")
    systolic_bp:   Optional[float] = Field(None, ge=0, le=300, description="mmHg")
    diastolic_bp:  Optional[float] = Field(None, ge=0, le=200, description="mmHg")
    resp_rate:     Optional[float] = Field(None, ge=0, le=80,  description="breaths/min")
    spo2:          Optional[float] = Field(None, ge=0, le=100, description="%")
    temperature:   Optional[float] = Field(None, ge=80, le=115, description="Fahrenheit")


class Comorbidities(BaseModel):
    has_diabetes:      int = Field(0, ge=0, le=1)
    has_hypertension:  int = Field(0, ge=0, le=1)
    has_heart_disease: int = Field(0, ge=0, le=1)
    has_sepsis:        int = Field(0, ge=0, le=1)
    has_resp_failure:  int = Field(0, ge=0, le=1)
    
class TriageRequest(BaseModel):
    age:              float          = Field(..., ge=0, le=120)
    gender:           Literal["M", "F"]
    vitals:           Vitals
    comorbidities:   Comorbidities = Field(default_factory=Comorbidities)
    complaint_cat:   int           = Field(0, ge=0, le=5,
                                          description="0=other,1=cardiac,2=infection,3=respiratory,4=neuro,5=abdominal")


class TriageResponse(BaseModel):
    model_config = {"protected_namespaces": ()}   # allow `model_version` field name

    esi_level:     int    = Field(..., ge=1, le=5, description="1=resuscitation, 5=non-urgent")
    confidence:    float  = Field(..., ge=0, le=1)
    top_risk_factors: list[str] = Field(
        default_factory=list,
        description="Top 3 SHAP-derived reasons for this score"
    )
    model_version: str


# ─── Queue ──────────────────────────────────────────────────────────────

class QueueEntry(BaseModel):
    patient_id:       str
    arrival_time:     datetime
    waiting_minutes:  int               = Field(..., ge=0, description="recomputed each request")
    esi_level:        int               = Field(..., ge=1, le=5)
    confidence:       float             = Field(..., ge=0, le=1)
    age:              float
    gender:           Literal["M", "F"]
    complaint_cat:    int               = Field(..., ge=0, le=5)
    complaint_label:  str
    top_risk_factors: list[str]         = Field(default_factory=list)


class QueueResponse(BaseModel):
    count:         int
    last_updated:  datetime
    patients:      list[QueueEntry]


# ─── Beds ───────────────────────────────────────────────────────────────

BedStatusLiteral = Literal["available", "occupied", "cleaning"]


class BedStatus(BaseModel):
    bed_id:      str
    category:    str           = Field(..., description="resuscitation, trauma, monitored, general, fast_track")
    status:      BedStatusLiteral
    patient_id:  Optional[str] = None
    since:       Optional[datetime] = Field(None, description="when current status started")


class BedCategorySummary(BaseModel):
    total:      int
    available:  int
    occupied:   int
    cleaning:   int


class BedsSummary(BaseModel):
    total:        int
    available:    int
    occupied:     int
    cleaning:     int
    by_category:  dict[str, BedCategorySummary]


class BedsResponse(BaseModel):
    last_updated:  datetime
    summary:       BedsSummary
    beds:          list[BedStatus]
