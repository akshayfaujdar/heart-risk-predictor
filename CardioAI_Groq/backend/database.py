"""
database.py
MySQL database schema for CardioAI using SQLAlchemy + PyMySQL.

Connection URL format:
    mysql+pymysql://USER:PASSWORD@HOST:PORT/DATABASE

Set DATABASE_URL in your .env file.
"""

import os
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float,
    ForeignKey, Integer, String, Text, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

# ---------------------------------------------------------------------------
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "mysql+pymysql://root:yourpassword@localhost:3306/cardioai_db"
)

engine       = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=3600)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

class User(Base):
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    email      = Column(String(255), unique=True, nullable=False, index=True)
    name       = Column(String(100))
    hashed_pw  = Column(String(255))
    is_active  = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    patients     = relationship("Patient",     back_populates="user")
    chat_history = relationship("ChatHistory", back_populates="user")


class Patient(Base):
    __tablename__ = "patients"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    user_id        = Column(Integer, ForeignKey("users.id"), nullable=True)
    name           = Column(String(100), default="Anonymous")
    age            = Column(Float)
    gender         = Column(Integer)        # 0 = Female, 1 = Male
    chest_pain     = Column(Integer, default=0)
    sob            = Column(Integer, default=0)
    fatigue        = Column(Integer, default=0)
    palpitations   = Column(Integer, default=0)
    dizziness      = Column(Integer, default=0)
    swelling       = Column(Integer, default=0)
    pain_arms      = Column(Integer, default=0)
    cold_sweats    = Column(Integer, default=0)
    high_bp        = Column(Integer, default=0)
    high_chol      = Column(Integer, default=0)
    diabetes       = Column(Integer, default=0)
    smoking        = Column(Integer, default=0)
    obesity        = Column(Integer, default=0)
    sedentary      = Column(Integer, default=0)
    family_hist    = Column(Integer, default=0)
    chronic_stress = Column(Integer, default=0)
    created_at     = Column(DateTime, default=datetime.utcnow)

    user        = relationship("User",       back_populates="patients")
    predictions = relationship("Prediction", back_populates="patient", cascade="all, delete")


class Prediction(Base):
    __tablename__ = "predictions"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    patient_id   = Column(Integer, ForeignKey("patients.id"))
    probability  = Column(Float)
    risk_percent = Column(Float)
    risk_level   = Column(String(20))   # Low / Moderate / High
    prediction   = Column(Integer)      # 0 or 1
    model_used   = Column(String(50),  default="Gradient Boosting")
    model_acc    = Column(Float,       default=0.9931)
    created_at   = Column(DateTime,    default=datetime.utcnow)

    patient = relationship("Patient", back_populates="predictions")


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=True)
    role       = Column(String(20))         # user | assistant
    content    = Column(Text)
    session_id = Column(String(100), default="default")
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="chat_history")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def init_db():
    """Create all tables in MySQL. Safe to call multiple times."""
    Base.metadata.create_all(bind=engine)
    print("MySQL tables created / verified.")


def get_db():
    """FastAPI dependency — yields one DB session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
