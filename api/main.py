"""
FastAPI entrypoint for the ER triage system.

Run locally:
    uvicorn api.main:app --reload

Skeleton only: /triage returns a hard-coded dummy response.
The real model + feature pipeline will replace the stub once trained.
"""
from fastapi import FastAPI

from api.schemas import TriageRequest, TriageResponse

app = FastAPI(
    title="ER Triage System",
    description="Predicts ESI (Emergency Severity Index 1-5) at patient arrival.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/triage", response_model=TriageResponse)
def triage(req: TriageRequest) -> TriageResponse:
    return TriageResponse(
        esi_level=3,
        confidence=0.0,
        reasoning="stub response — model not yet trained",
        model_version="0.1.0-dummy",
    )
