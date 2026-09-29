"""SQLAlchemy 2.0 ORM models for VeriTrust AI."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


def utc_now() -> datetime:
    """Return current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class Request(Base):
    """Customer request and full compliance audit log record."""

    __tablename__ = "requests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_evidence: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON, nullable=True)
    maker_output: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    judge_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    approved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="PENDING", nullable=False)  # APPROVED, BLOCKED, PENDING, ERROR
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    final_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships
    hallucinations: Mapped[List["Hallucination"]] = relationship(
        back_populates="request",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    metric: Mapped[Optional["Metric"]] = relationship(
        back_populates="request",
        cascade="all, delete-orphan",
        lazy="selectin",
        uselist=False,
    )
    audit_traces: Mapped[List["AuditTrace"]] = relationship(
        back_populates="request",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="AuditTrace.step_index",
    )

    def __repr__(self) -> str:
        return f"<Request(id='{self.id}', status='{self.status}', approved={self.approved})>"


class Hallucination(Base):
    """Specific hallucination or factual non-compliance issue detected by the Judge."""

    __tablename__ = "hallucinations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("requests.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    iteration: Mapped[int] = mapped_column(Integer, default=0, nullable=False)  # 0: initial draft, 1-3: corrections
    issue_type: Mapped[str] = mapped_column(String(50), nullable=False)  # CONTRADICTION, FABRICATED_POLICY, etc.
    description: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(String(20), default="HIGH", nullable=False)  # CRITICAL, HIGH, MEDIUM, LOW
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_ref: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationship
    request: Mapped["Request"] = relationship(back_populates="hallucinations")

    def __repr__(self) -> str:
        return f"<Hallucination(id={self.id}, type='{self.issue_type}', severity='{self.severity}')>"


class Metric(Base):
    """Aggregated quantitative performance and accuracy metrics per request."""

    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("requests.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    accuracy_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    approval_score: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    hallucination_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationship
    request: Mapped["Request"] = relationship(back_populates="metric")

    def __repr__(self) -> str:
        return f"<Metric(id={self.id}, request_id='{self.request_id}', latency={self.latency_ms:.1f}ms)>"


class AuditTrace(Base):
    """Sequential execution step record for full explainability and live monitoring."""

    __tablename__ = "audit_traces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("requests.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    step_name: Mapped[str] = mapped_column(String(50), nullable=False)  # RETRIEVAL, MAKER, JUDGE_0, CORRECTOR_1, etc.
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationship
    request: Mapped["Request"] = relationship(back_populates="audit_traces")

    def __repr__(self) -> str:
        return f"<AuditTrace(step={self.step_index}, name='{self.step_name}')>"
