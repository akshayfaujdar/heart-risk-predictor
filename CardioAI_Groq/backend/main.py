"""
main.py
FastAPI REST API for CardioAI Heart Disease Analysis System.

Run:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Swagger docs:
    http://localhost:8000/docs
"""

import json
import os
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Load .env file
load_dotenv()

from chatbot import explain_prediction, get_chatbot_response
from database import ChatHistory, Patient, Prediction, get_db, init_db
from model import load_model, predict_risk, train_and_evaluate

# ---------------------------------------------------------------------------
app = FastAPI(
    title="CardioAI API",
    description="Heart Disease Risk Analysis & Prevention System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model + scaler (loaded once at startup)
_model  = None
_scaler = None

DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "dataset",
    "heart_disease_risk_dataset_earlymed.csv"
)

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    global _model, _scaler
    try:
        _model, _scaler = load_model()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("No saved model found. Training now (this may take 1-2 minutes)...")
        train_and_evaluate(DATASET_PATH)
        _model, _scaler = load_model()
        print("Training complete.")
    init_db()

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PatientInput(BaseModel):
    Age: float
    Gender: int = 0                  # 0 = Female, 1 = Male
    Chest_Pain: int = 0
    Shortness_of_Breath: int = 0
    Fatigue: int = 0
    Palpitations: int = 0
    Dizziness: int = 0
    Swelling: int = 0
    Pain_Arms_Jaw_Back: int = 0
    Cold_Sweats_Nausea: int = 0
    High_BP: int = 0
    High_Cholesterol: int = 0
    Diabetes: int = 0
    Smoking: int = 0
    Obesity: int = 0
    Sedentary_Lifestyle: int = 0
    Family_History: int = 0
    Chronic_Stress: int = 0


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    history: Optional[List[dict]] = []


class PatientSaveRequest(BaseModel):
    patient: PatientInput
    name: Optional[str] = "Anonymous"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "status":    "CardioAI API is running",
        "version":   "1.0.0",
        "model":     "Gradient Boosting",
        "accuracy":  "99.31%",
        "docs":      "/docs",
    }


# ── POST /predict ────────────────────────────────────────────────────────────

@app.post("/predict")
async def predict(data: PatientInput):
    """
    Predict heart disease risk from patient clinical data.

    Input  : 18 clinical features (symptoms + risk factors)
    Output : probability, risk_percent, prediction (0/1), risk_level
    """
    patient_dict = data.dict()
    result = predict_risk(patient_dict, model=_model, scaler=_scaler)

    return {
        "success":        True,
        "prediction":     result,
        "model":          "Gradient Boosting",
        "model_accuracy": 0.9931,
        "timestamp":      datetime.utcnow().isoformat(),
    }


# ── POST /chat ───────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest, db: Session = Depends(get_db)):
    """
    AI health chatbot powered by Groq (llama-3.3-70b-versatile).

    Input  : user message + optional conversation history
    Output : AI response text
    """
    reply = get_chatbot_response(req.message, req.history)

    # Save conversation to MySQL
    db.add(ChatHistory(role="user",      content=req.message, session_id=req.session_id))
    db.add(ChatHistory(role="assistant", content=reply,        session_id=req.session_id))
    db.commit()

    return {
        "success":    True,
        "response":   reply,
        "session_id": req.session_id,
    }


# ── GET /analysis ────────────────────────────────────────────────────────────

@app.get("/analysis")
async def analysis():
    """
    Return dataset statistics and model performance metrics.
    Run 'python model.py' first to generate model_results.json.
    """
    results_path = os.path.join(os.path.dirname(__file__), "model_results.json")

    if not os.path.exists(results_path):
        raise HTTPException(
            status_code=404,
            detail="model_results.json not found. Run: python model.py"
        )

    with open(results_path) as f:
        return json.load(f)


# ── POST /patient/save ───────────────────────────────────────────────────────

@app.post("/patient/save")
async def save_patient(req: PatientSaveRequest, db: Session = Depends(get_db)):
    """
    Save patient record to MySQL and return the prediction result.
    """
    d = req.patient.dict()

    patient = Patient(
        name           = req.name,
        age            = d["Age"],
        gender         = d["Gender"],
        chest_pain     = d["Chest_Pain"],
        sob            = d["Shortness_of_Breath"],
        fatigue        = d["Fatigue"],
        palpitations   = d["Palpitations"],
        dizziness      = d["Dizziness"],
        swelling       = d["Swelling"],
        pain_arms      = d["Pain_Arms_Jaw_Back"],
        cold_sweats    = d["Cold_Sweats_Nausea"],
        high_bp        = d["High_BP"],
        high_chol      = d["High_Cholesterol"],
        diabetes       = d["Diabetes"],
        smoking        = d["Smoking"],
        obesity        = d["Obesity"],
        sedentary      = d["Sedentary_Lifestyle"],
        family_hist    = d["Family_History"],
        chronic_stress = d["Chronic_Stress"],
    )
    db.add(patient)
    db.flush()  # get patient.id before commit

    result = predict_risk(d, model=_model, scaler=_scaler)

    pred = Prediction(
        patient_id   = patient.id,
        probability  = result["probability"],
        risk_percent = result["risk_percent"],
        risk_level   = result["risk_level"],
        prediction   = result["prediction"],
    )
    db.add(pred)
    db.commit()

    return {
        "success":    True,
        "patient_id": patient.id,
        "prediction": result,
    }


# ── GET /patient/history ─────────────────────────────────────────────────────

@app.get("/patient/history")
async def patient_history(limit: int = 50, db: Session = Depends(get_db)):
    """
    Return the most recent patient prediction records from MySQL.
    """
    rows = (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .limit(limit)
        .all()
    )

    return {
        "success": True,
        "count":   len(rows),
        "predictions": [
            {
                "id":           row.id,
                "patient_id":   row.patient_id,
                "risk_level":   row.risk_level,
                "risk_percent": row.risk_percent,
                "model_used":   row.model_used,
                "created_at":   row.created_at.isoformat(),
            }
            for row in rows
        ],
    }
