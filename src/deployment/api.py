"""
FastAPI service for real-time MP recommendations.

Endpoints:
    POST /recommend      — Get an MP recommendation for a patient
    GET  /health         — Health check
    GET  /model/info     — Model metadata
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from src.deployment.safety_filter import SafetyFilter
from src.deployment.explainer import ExplanationGenerator


# ===================================================================
# Request / Response schemas
# ===================================================================
class PatientState(BaseModel):
    """Input: current patient state for recommendation."""

    patient_id: str
    # Ventilator
    mechanical_power: float = Field(..., ge=0, le=100)
    tidal_volume: float = Field(..., ge=50, le=1500)
    respiratory_rate: float = Field(..., ge=5, le=50)
    peep: float = Field(..., ge=0, le=30)
    fio2: float = Field(..., ge=0.21, le=1.0)
    plateau_pressure: Optional[float] = None
    peak_pressure: Optional[float] = None
    driving_pressure: Optional[float] = None
    compliance: Optional[float] = None
    # Vitals
    spo2: float = Field(..., ge=50, le=100)
    heart_rate: float = Field(80, ge=20, le=250)
    mean_arterial_pressure: float = Field(75, ge=20, le=200)
    temperature: float = Field(37.0)
    # Labs (optional)
    pao2: Optional[float] = None
    paco2: Optional[float] = None
    pf_ratio: Optional[float] = None
    lactate: Optional[float] = None
    ph: Optional[float] = None
    # Context
    hours_on_ventilator: int = 0
    age: Optional[int] = None


class Recommendation(BaseModel):
    """Output: the recommendation with explanation and evidence."""

    patient_id: str
    timestamp: str
    action: str
    action_index: int
    target_mp: float
    confidence: float
    explanation: str
    safety_alerts: list[str]
    was_overridden: bool


# ===================================================================
# Application
# ===================================================================
def create_app(config: dict, model_path: str | Path) -> "FastAPI":
    """
    Build and return the FastAPI application with a loaded model.
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("fastapi and uvicorn are required. pip install fastapi uvicorn")

    from src.models.cql_agent import MPAdvisorCQL

    app = FastAPI(
        title="MP Advisor API",
        description="Personalised Mechanical Power recommendations for ICU patients",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Load model and components
    agent = MPAdvisorCQL(config)
    agent.load(model_path)
    safety = SafetyFilter(config)
    explainer = ExplanationGenerator()

    ACTION_NAMES = MPAdvisorCQL.ACTION_NAMES
    ACTION_DELTAS = MPAdvisorCQL.ACTION_DELTAS

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/model/info")
    def model_info():
        return {
            "model": "CQL (Conservative Q-Learning)",
            "version": config["project"]["version"],
            "actions": ACTION_NAMES,
        }

    @app.post("/recommend", response_model=Recommendation)
    def recommend(patient: PatientState):
        # Build state dict for safety filter
        state_dict = patient.model_dump()

        # Build numeric state vector (simplified — real version uses full feature pipeline)
        state_vector = np.array(
            [
                patient.mechanical_power,
                patient.tidal_volume,
                patient.respiratory_rate,
                patient.peep,
                patient.fio2,
                patient.spo2,
                patient.heart_rate,
                patient.mean_arterial_pressure,
                patient.pao2 or 0,
                patient.paco2 or 0,
                patient.lactate or 0,
                patient.hours_on_ventilator,
                patient.age or 60,
            ],
            dtype=np.float32,
        )

        # Get raw recommendation
        result = agent.predict(state_vector)
        raw_action = result["action"]

        # Safety filter
        safe_action, alerts = safety.filter(state_dict, raw_action)
        alerts += safety.check_alerts(state_dict)

        # Explanation
        explanation = explainer.generate(
            patient_state=state_dict,
            action=safe_action,
            confidence=result["confidence"],
            q_values=result.get("q_values"),
        )

        target_mp = patient.mechanical_power + ACTION_DELTAS[safe_action]

        return Recommendation(
            patient_id=patient.patient_id,
            timestamp=datetime.utcnow().isoformat(),
            action=ACTION_NAMES[safe_action],
            action_index=safe_action,
            target_mp=round(target_mp, 1),
            confidence=round(result["confidence"], 3),
            explanation=explanation,
            safety_alerts=alerts,
            was_overridden=safe_action != raw_action,
        )

    return app
