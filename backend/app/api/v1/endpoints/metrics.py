"""API endpoints for dashboard metrics and audit history."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.session import get_db
from backend.app.db.models import AuditTrace, Hallucination, Metric, Request

router = APIRouter(prefix="/api/v1", tags=["Dashboard Metrics"])


class DashboardMetrics(BaseModel):
    """Aggregated metrics for the VeriTrust dashboard."""
    total_requests: int
    approved_count: int
    blocked_count: int
    corrected_count: int  # requests that needed >=1 correction attempt
    avg_latency_ms: Optional[float]
    avg_judge_score: Optional[float]
    total_hallucinations_detected: int
    approval_rate: float  # 0.0–1.0
    avg_correction_attempts: float
    recent_requests: List[Dict[str, Any]]


class AuditTraceResponse(BaseModel):
    """Full audit trail for a single request."""
    request_id: str
    question: str
    status: str
    approved: bool
    final_answer: Optional[str]
    judge_score: Optional[int]
    retry_count: int
    latency_ms: Optional[float]
    maker_output: Optional[Dict[str, Any]]
    retrieved_evidence: Optional[List[Dict[str, Any]]]
    audit_traces: List[Dict[str, Any]]
    hallucinations: List[Dict[str, Any]]


@router.get(
    "/metrics",
    response_model=DashboardMetrics,
    summary="Get Aggregated Dashboard Metrics",
    description="Returns compliance statistics derived from the audit database.",
)
async def get_dashboard_metrics(db: AsyncSession = Depends(get_db)):
    # Total requests
    total_result = await db.execute(select(func.count(Request.id)))
    total = total_result.scalar() or 0

    # Approved / blocked counts
    approved_result = await db.execute(
        select(func.count(Request.id)).where(Request.approved == True)
    )
    approved = approved_result.scalar() or 0
    blocked = total - approved

    # Corrected (retry_count >= 1)
    corrected_result = await db.execute(
        select(func.count(Request.id)).where(Request.retry_count >= 1)
    )
    corrected = corrected_result.scalar() or 0

    # Average latency
    avg_latency_result = await db.execute(select(func.avg(Metric.latency_ms)))
    avg_latency = avg_latency_result.scalar()

    # Average judge score
    avg_score_result = await db.execute(
        select(func.avg(Request.judge_score)).where(Request.judge_score != None)
    )
    avg_score = avg_score_result.scalar()

    # Total hallucinations
    halluc_result = await db.execute(select(func.count(Hallucination.id)))
    total_hallucinations = halluc_result.scalar() or 0

    # Average correction attempts
    avg_retry_result = await db.execute(select(func.avg(Metric.retry_count)))
    avg_retry = avg_retry_result.scalar() or 0.0

    # Recent 20 requests for the history table
    recent_result = await db.execute(
        select(Request).order_by(Request.created_at.desc()).limit(20)
    )
    recent_rows = recent_result.scalars().all()
    recent_requests = [
        {
            "id": r.id,
            "question": r.question[:80] + ("…" if len(r.question) > 80 else ""),
            "status": r.status,
            "approved": r.approved,
            "judge_score": r.judge_score,
            "retry_count": r.retry_count,
            "latency_ms": r.latency_ms,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in recent_rows
    ]

    return DashboardMetrics(
        total_requests=total,
        approved_count=approved,
        blocked_count=blocked,
        corrected_count=corrected,
        avg_latency_ms=round(float(avg_latency), 1) if avg_latency is not None else None,
        avg_judge_score=round(float(avg_score), 1) if avg_score is not None else None,
        total_hallucinations_detected=total_hallucinations,
        approval_rate=round(approved / total, 3) if total > 0 else 0.0,
        avg_correction_attempts=round(float(avg_retry), 2) if avg_retry else 0.0,
        recent_requests=recent_requests,
    )


@router.get(
    "/audit/{request_id}",
    response_model=AuditTraceResponse,
    summary="Get Full Audit Trail for a Request",
    description="Returns the complete step-by-step execution trace for a given request ID.",
)
async def get_audit_trace(request_id: str, db: AsyncSession = Depends(get_db)):
    from fastapi import HTTPException, status

    result = await db.execute(select(Request).where(Request.id == request_id))
    req = result.scalar_one_or_none()
    if req is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Request '{request_id}' not found.",
        )

    traces = [
        {
            "step_index": t.step_index,
            "step_name": t.step_name,
            "payload": t.payload,
            "timestamp": t.timestamp.isoformat() if t.timestamp else None,
        }
        for t in req.audit_traces
    ]

    hallucinations = [
        {
            "id": h.id,
            "iteration": h.iteration,
            "issue_type": h.issue_type,
            "description": h.description,
            "severity": h.severity,
            "claim_text": h.claim_text,
            "evidence_ref": h.evidence_ref,
        }
        for h in req.hallucinations
    ]

    return AuditTraceResponse(
        request_id=req.id,
        question=req.question,
        status=req.status,
        approved=req.approved,
        final_answer=req.final_answer,
        judge_score=req.judge_score,
        retry_count=req.retry_count,
        latency_ms=req.latency_ms,
        maker_output=req.maker_output,
        retrieved_evidence=req.retrieved_evidence,
        audit_traces=traces,
        hallucinations=hallucinations,
    )
