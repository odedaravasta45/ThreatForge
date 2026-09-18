"""
SQLAlchemy ORM models for ThreatLens AI.
All tables are created in SQLite via create_all on startup.
"""
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey, JSON
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class AlertModel(Base):
    __tablename__ = "alerts"

    id = Column(String, primary_key=True, index=True)
    timestamp = Column(String, nullable=False)
    source = Column(String, nullable=False)          # SIEM | EDR | Network Sensor | Threat Intel
    event = Column(String, nullable=False)
    src_ip = Column(String, default="unknown")
    dest_ip = Column(String, default="unknown")
    host = Column(String, default="unknown")
    severity = Column(String, nullable=False)         # CRITICAL|HIGH|MEDIUM|LOW|INFORMATIONAL
    status = Column(String, default="UNPROCESSED")    # CORRELATED|ANALYZED|UNPROCESSED|FALSE_POSITIVE
    correlation_id = Column(String, ForeignKey("incidents.id", use_alter=True, name="fk_alert_incident"), nullable=True, index=True)
    raw_event = Column(Text, nullable=False)
    normalized_event = Column(Text, nullable=False)
    mitre_techniques = Column(JSON, default=list)
    # Per-alert ML inference (filled at ingestion time).
    ml_model = Column(String, default="XGBClassifier")
    ml_threat_probability = Column(Integer, default=0)
    ml_classification = Column(String, default="INVESTIGATING")
    ml_priority = Column(String, default="MEDIUM")
    created_at = Column(DateTime, default=datetime.utcnow)


class IncidentModel(Base):
    __tablename__ = "incidents"

    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    priority = Column(String, nullable=False)         # CRITICAL|HIGH|MEDIUM|LOW
    confidence = Column(Integer, nullable=False)      # 0–99
    threat_score = Column(Integer, default=0)
    related_alerts_count = Column(Integer, default=0)
    mitre_techniques = Column(JSON, default=list)
    status = Column(String, default="NEW")            # NEW|INVESTIGATING|MONITORING|RESOLVED
    classification = Column(String, default="INVESTIGATING")  # GENUINE_THREAT|LIKELY_FALSE_POSITIVE|INVESTIGATING
    first_seen = Column(String, nullable=False)
    last_seen = Column(String, nullable=False)
    affected_assets = Column(JSON, default=list)
    why_correlated = Column(Text, default="")
    timeline = Column(JSON, default=list)
    source_count = Column(Integer, default=1)
    ml_model = Column(String, default="XGBClassifier")
    ml_threat_probability = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class MitreTechniqueModel(Base):
    __tablename__ = "mitre_techniques"

    id = Column(String, primary_key=True)             # T1059.001
    name = Column(String, nullable=False)
    tactic = Column(String, nullable=False)
    count = Column(Integer, default=0)
    severity = Column(String, default="MEDIUM")
    last_observed = Column(String, default="")
    description = Column(Text, default="")
