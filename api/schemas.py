"""Request / response schemas for the triage API."""
from typing import Optional, Literal
from pydantic import BaseModel, Field


class Vitals(BaseModel):
    heart_rate: Optional[float] = Field(None, ge=0, le=300, description="bpm")
    sbp:        Optional[float] = Field(None, ge=0, le=300, description="systolic BP, mmHg")
    dbp:        Optional[float] = Field(None, ge=0, le=200, description="diastolic BP, mmHg")
    resp_rate:  Optional[float] = Field(None, ge=0, le=80,  description="breaths/min")
    spo2:       Optional[float] = Field(None, ge=0, le=100, description="%")
    temp_c:     Optional[float] = Field(None, ge=25, le=45, description="Celsius")
    gcs_total:  Optional[int]   = Field(None, ge=3, le=15,  description="Glasgow Coma Scale 3-15")


class TriageRequest(BaseModel):
    age:              float          = Field(..., ge=0, le=120)
    gender:           Literal["M", "F"]
    vitals:           Vitals
    icd9_codes:       list[str]      = Field(default_factory=list, description="patient history ICD-9 codes")
    chief_complaint:  str            = Field("", description="free-text complaint at arrival")


class TriageResponse(BaseModel):
    esi_level:     int    = Field(..., ge=1, le=5, description="1=resuscitation, 5=non-urgent")
    confidence:    float  = Field(..., ge=0, le=1)
    reasoning:     str
    model_version: str
