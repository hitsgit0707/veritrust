"""API endpoint for independent testing of the Maker Agent."""

from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.app.agents.maker import MakerAgent, MakerResult

router = APIRouter(prefix="/api/v1/maker", tags=["Maker Agent"])


class MakerDraftRequest(BaseModel):
    """Request payload for generating a Maker draft."""
    question: str = Field(..., min_length=1, max_length=1000, description="Customer question")
    top_k: Optional[int] = Field(default=4, ge=1, le=10, description="Number of evidence chunks to retrieve")
    adversarial_mode: Optional[str] = Field(default=None, description="Optional adversarial test trigger")


@router.post(
    "/draft",
    response_model=MakerResult,
    summary="Generate Grounded Maker Draft Answer",
    description="Invokes RAG retrieval and produces a Maker draft answer citing verified evidence.",
)
async def create_maker_draft(payload: MakerDraftRequest):
    maker = MakerAgent()
    result = await maker.generate_draft(
        question=payload.question,
        top_k=payload.top_k or 4,
        adversarial_mode=payload.adversarial_mode,
    )
    return result
