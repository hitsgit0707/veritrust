"""API endpoint for independent testing of the Judge Agent."""

from typing import List, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.app.agents.judge import JudgeAgent, JudgeResult
from backend.app.rag.vector_store import RetrievedChunk

router = APIRouter(prefix="/api/v1/judge", tags=["Judge Agent"])


class JudgeEvaluateRequest(BaseModel):
    """Request payload for evaluating a Maker draft."""
    question: str = Field(..., min_length=1, max_length=2000, description="Customer question")
    draft_answer: str = Field(..., min_length=1, description="Maker draft answer to evaluate")
    sources: Optional[List[str]] = Field(default_factory=list, description="Maker cited source identifiers")
    confidence: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Maker self-assessed confidence")
    retrieved_evidence: Optional[List[RetrievedChunk]] = Field(default_factory=list, description="Retrieved evidence chunks")


@router.post(
    "",
    response_model=JudgeResult,
    summary="Evaluate Draft Answer with Judge Agent",
    description="Validates a Maker draft against verified knowledge-base evidence.",
)
@router.post(
    "/evaluate",
    response_model=JudgeResult,
    summary="Evaluate Draft Answer with Judge Agent",
    description="Validates a Maker draft against verified knowledge-base evidence.",
)
async def evaluate_maker_draft(payload: JudgeEvaluateRequest):
    judge = JudgeAgent()
    result = await judge.evaluate(
        question=payload.question,
        draft_answer=payload.draft_answer,
        retrieved_evidence=payload.retrieved_evidence or [],
        sources=payload.sources,
        confidence=payload.confidence,
    )
    return result
