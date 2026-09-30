"""API endpoint for customer-facing compliance workflow query."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.session import get_db
from backend.app.graph.workflow import run_compliance_workflow
from backend.app.schemas.compliance import AskRequest, AskResponse

router = APIRouter(prefix="/api/v1", tags=["Compliance Workflow"])


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Submit Question to Compliance Filter",
    description="Executes the full LangGraph Maker-Judge-Correction workflow and returns verified answer.",
)
async def ask_compliance_filter(
    payload: AskRequest,
    db: AsyncSession = Depends(get_db),
):
    final_state = await run_compliance_workflow(
        question=payload.question,
        metadata=payload.metadata,
        db_session=db,
    )

    judge_dict = final_state.get("judge_result") or {}
    judge_score = judge_dict.get("score")
    is_approved = (final_state.get("status") == "APPROVED")

    # Format sources for response
    sources = final_state.get("sources", []) if is_approved else []

    # Extract issues
    issues = final_state.get("all_issues", [])

    return AskResponse(
        request_id=final_state.get("request_id", ""),
        question=final_state.get("question", payload.question),
        status=final_state.get("status", "BLOCKED"),
        approved=is_approved,
        answer=final_state.get("final_answer", ""),
        sources=sources,
        judge_score=judge_score,
        correction_attempts=final_state.get("correction_attempts", 0),
        latency_ms=final_state.get("latency_ms"),
        issues=issues,
    )
