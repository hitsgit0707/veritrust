"""LangGraph Compliance Workflow orchestration for VeriTrust AI."""

from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional, TypedDict
import uuid

from langgraph.graph import END, START, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.agents.corrector import CorrectionAgent, CorrectionResult
from backend.app.agents.judge import JudgeAgent, JudgeResult
from backend.app.agents.maker import MakerAgent, MakerResult
from backend.app.core.llm import BaseLLMProvider, get_llm_provider
from backend.app.db.models import AuditTrace, Hallucination, Metric, Request
from backend.app.rag.service import RAGService, get_rag_service
from backend.app.rag.vector_store import RetrievedChunk

SAFE_FALLBACK_ANSWER = "I'm unable to provide a verified answer from the available company knowledge base."


class WorkflowState(TypedDict, total=False):
    """Explicit typed state for LangGraph compliance filter workflow."""
    request_id: str
    question: str
    top_k: int
    adversarial_mode: Optional[str]
    metadata: Optional[Dict[str, Any]]

    # Step outputs
    retrieved_evidence: List[RetrievedChunk]
    maker_draft: str
    maker_confidence: float
    current_answer: str
    sources: List[str]
    judge_result: Optional[Dict[str, Any]]
    correction_result: Optional[Dict[str, Any]]
    correction_attempts: int
    final_answer: str
    status: str  # APPROVED, BLOCKED, PENDING
    all_issues: List[Dict[str, Any]]

    # Audit & telemetry
    audit_trail: List[Dict[str, Any]]
    latency_ms: float


def _record_audit_step(state: WorkflowState, step_name: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Helper to append an immutable audit trace item to the state's audit trail."""
    trail = list(state.get("audit_trail", []))
    step_index = len(trail)
    trail.append({
        "step_index": step_index,
        "step_name": step_name,
        "payload": payload,
        "timestamp": datetime.now(timezone.utc),
    })
    return trail


def create_compliance_workflow(
    llm_provider: Optional[BaseLLMProvider] = None,
    rag_service: Optional[RAGService] = None,
):
    """
    Construct and compile the LangGraph compliance filter workflow.
    
    Graph topology:
    START -> retrieval -> maker -> judge <────────┐
                                     │ (conditional)
                                     ├─ approved ─> finalize ─> END
                                     ├─ rejected (attempts < 3) ─> correction ──┘
                                     └─ rejected (attempts >= 3) ─> block ─> END
    """
    llm = llm_provider or get_llm_provider()
    rag = rag_service or get_rag_service()

    maker_agent = MakerAgent(llm_provider=llm, rag_service=rag)
    judge_agent = JudgeAgent(llm_provider=llm)
    corrector_agent = CorrectionAgent(llm_provider=llm)

    # --------------------------------------------------------------------------
    # Node Definitions
    # --------------------------------------------------------------------------

    async def retrieval_node(state: WorkflowState) -> Dict[str, Any]:
        """Node 1: Retrieve grounded knowledge chunks if not pre-populated."""
        evidence = state.get("retrieved_evidence")
        if evidence is None:
            top_k = state.get("top_k", 4)
            evidence = rag.retrieve(query=state["question"], top_k=top_k)

        trail = _record_audit_step(
            state,
            "RETRIEVAL",
            {
                "question": state["question"],
                "chunks_retrieved": len(evidence),
                "sources": [c.source for c in evidence],
            },
        )
        return {
            "retrieved_evidence": evidence,
            "audit_trail": trail,
        }

    async def maker_node(state: WorkflowState) -> Dict[str, Any]:
        """Node 2: Formulate initial draft response citing evidence."""
        evidence = state.get("retrieved_evidence", [])
        adv_mode = state.get("adversarial_mode")

        maker_res: MakerResult = await maker_agent.generate_draft(
            question=state["question"],
            top_k=state.get("top_k", 4),
            adversarial_mode=adv_mode,
        )

        trail = _record_audit_step(
            state,
            "MAKER",
            {
                "draft_answer": maker_res.draft_answer,
                "confidence": maker_res.confidence,
                "sources": maker_res.sources,
            },
        )

        return {
            "maker_draft": maker_res.draft_answer,
            "maker_confidence": maker_res.confidence,
            "current_answer": maker_res.draft_answer,
            "sources": maker_res.sources,
            "retrieved_evidence": maker_res.retrieved_evidence or evidence,
            "correction_attempts": 0,
            "all_issues": [],
            "audit_trail": trail,
        }

    async def judge_node(state: WorkflowState) -> Dict[str, Any]:
        """Node 3: Independently audit current answer against retrieved evidence."""
        attempts = state.get("correction_attempts", 0)
        judge_res: JudgeResult = await judge_agent.evaluate(
            question=state["question"],
            draft_answer=state["current_answer"],
            retrieved_evidence=state.get("retrieved_evidence", []),
            sources=state.get("sources", []),
            confidence=state.get("maker_confidence", 1.0),
        )

        # Accumulate detected issues across all attempts
        accumulated_issues = list(state.get("all_issues", []))
        for iss in judge_res.issues:
            iss_dict = iss.model_dump()
            iss_dict["iteration"] = attempts
            accumulated_issues.append(iss_dict)

        step_name = f"JUDGE_{attempts}"
        trail = _record_audit_step(
            state,
            step_name,
            {
                "iteration": attempts,
                "approved": judge_res.approved,
                "score": judge_res.score,
                "issues_count": len(judge_res.issues),
                "issues": [i.model_dump() for i in judge_res.issues],
            },
        )

        return {
            "judge_result": judge_res.model_dump(),
            "all_issues": accumulated_issues,
            "audit_trail": trail,
        }

    def router_after_judge(state: WorkflowState) -> str:
        """Conditional routing based on Judge verdict and retry threshold."""
        judge_dict = state.get("judge_result") or {}
        approved = judge_dict.get("approved", False)
        attempts = state.get("correction_attempts", 0)

        if approved:
            return "finalize"
        if attempts >= 3:
            return "block"
        return "correct"

    async def correction_node(state: WorkflowState) -> Dict[str, Any]:
        """Node 4: Rewrite rejected draft to remove hallucinations."""
        attempts = state.get("correction_attempts", 0) + 1
        judge_dict = state.get("judge_result", {})
        evidence = state.get("retrieved_evidence", [])

        correction_res: CorrectionResult = await corrector_agent.correct(
            question=state["question"],
            draft_answer=state["current_answer"],
            judge_result=judge_dict,
            retrieved_evidence=evidence,
        )

        step_name = f"CORRECTOR_{attempts}"
        trail = _record_audit_step(
            state,
            step_name,
            {
                "attempt": attempts,
                "corrected_answer": correction_res.corrected_answer,
                "explanation": correction_res.explanation,
                "corrections": correction_res.corrections,
            },
        )

        return {
            "current_answer": correction_res.corrected_answer,
            "correction_result": correction_res.model_dump(),
            "correction_attempts": attempts,
            "audit_trail": trail,
        }

    async def finalize_node(state: WorkflowState) -> Dict[str, Any]:
        """Node 5: Successfully finalize approved response."""
        # Safety invariant check
        judge_dict = state.get("judge_result", {})
        if not judge_dict.get("approved", False):
            # Fallback defensively if somehow reached without approval
            trail = _record_audit_step(state, "SAFETY_INTERCEPTION", {"reason": "Unapproved draft reached finalize"})
            return {
                "status": "BLOCKED",
                "final_answer": SAFE_FALLBACK_ANSWER,
                "audit_trail": trail,
            }

        trail = _record_audit_step(
            state,
            "FINALIZE",
            {
                "status": "APPROVED",
                "final_answer": state["current_answer"],
                "attempts": state.get("correction_attempts", 0),
            },
        )
        return {
            "status": "APPROVED",
            "final_answer": state["current_answer"],
            "audit_trail": trail,
        }

    async def block_node(state: WorkflowState) -> Dict[str, Any]:
        """Node 6: Enforce maximum retry blocking and return safe fallback."""
        trail = _record_audit_step(
            state,
            "BLOCK",
            {
                "status": "BLOCKED",
                "reason": "Exceeded maximum correction attempts (3) without Judge approval",
                "final_fallback": SAFE_FALLBACK_ANSWER,
                "attempts": state.get("correction_attempts", 3),
            },
        )
        return {
            "status": "BLOCKED",
            "final_answer": SAFE_FALLBACK_ANSWER,
            "audit_trail": trail,
        }

    # --------------------------------------------------------------------------
    # Graph Wiring
    # --------------------------------------------------------------------------
    workflow = StateGraph(WorkflowState)

    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("maker", maker_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("correction", correction_node)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("block", block_node)

    workflow.add_edge(START, "retrieval")
    workflow.add_edge("retrieval", "maker")
    workflow.add_edge("maker", "judge")

    workflow.add_conditional_edges(
        "judge",
        router_after_judge,
        {
            "finalize": "finalize",
            "correct": "correction",
            "block": "block",
        },
    )

    workflow.add_edge("correction", "judge")
    workflow.add_edge("finalize", END)
    workflow.add_edge("block", END)

    return workflow.compile()


async def run_compliance_workflow(
    question: str,
    request_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    llm_provider: Optional[BaseLLMProvider] = None,
    rag_service: Optional[RAGService] = None,
    db_session: Optional[AsyncSession] = None,
    top_k: int = 4,
    adversarial_mode: Optional[str] = None,
) -> WorkflowState:
    """
    Execute the full compliance pipeline and optionally persist to SQLite.
    
    Returns the final WorkflowState containing full audit trace and status.
    """
    start_time = time.perf_counter()
    req_id = request_id or str(uuid.uuid4())

    app = create_compliance_workflow(llm_provider=llm_provider, rag_service=rag_service)

    initial_state: WorkflowState = {
        "request_id": req_id,
        "question": question,
        "top_k": top_k,
        "adversarial_mode": adversarial_mode or (metadata.get("adversarial_mode") if metadata else None),
        "metadata": metadata or {},
        "status": "PENDING",
        "audit_trail": [],
        "correction_attempts": 0,
    }

    final_state = await app.ainvoke(initial_state)

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    final_state["latency_ms"] = latency_ms

    # Optional database persistence
    if db_session is not None:
        await persist_workflow_to_db(db_session, final_state)

    return final_state


async def persist_workflow_to_db(db: AsyncSession, state: WorkflowState) -> Request:
    """Save execution audit log, issues, and metrics into database tables."""
    req_id = state.get("request_id") or str(uuid.uuid4())
    judge_dict = state.get("judge_result") or {}
    judge_score = judge_dict.get("score")
    is_approved = (state.get("status") == "APPROVED")

    evidence_json = []
    for c in state.get("retrieved_evidence", []):
        if hasattr(c, "model_dump"):
            evidence_json.append(c.model_dump())
        elif isinstance(c, dict):
            evidence_json.append(c)

    maker_output_json = {
        "draft_answer": state.get("maker_draft", ""),
        "confidence": state.get("maker_confidence", 0.0),
        "sources": state.get("sources", []),
    }

    req_record = Request(
        id=req_id,
        question=state.get("question", ""),
        retrieved_evidence=evidence_json,
        maker_output=maker_output_json,
        judge_score=judge_score,
        approved=is_approved,
        status=state.get("status", "PENDING"),
        retry_count=state.get("correction_attempts", 0),
        final_answer=state.get("final_answer", ""),
        latency_ms=state.get("latency_ms", 0.0),
    )
    db.add(req_record)
    await db.flush()

    # Persist detected hallucinations/issues
    for iss in state.get("all_issues", []):
        hallucination = Hallucination(
            request_id=req_id,
            iteration=iss.get("iteration", 0),
            issue_type=iss.get("issue_type", "GENERAL"),
            description=iss.get("description", ""),
            severity=iss.get("severity", "HIGH"),
            claim_text=iss.get("claim_text", iss.get("claim", "")),
            evidence_ref=iss.get("evidence_ref", iss.get("evidence", "NONE")),
        )
        db.add(hallucination)

    # Persist Metric
    metric = Metric(
        request_id=req_id,
        latency_ms=state.get("latency_ms", 0.0),
        accuracy_score=float(judge_score or 0),
        approval_score=judge_score or 0,
        retry_count=state.get("correction_attempts", 0),
        hallucination_count=len(state.get("all_issues", [])),
    )
    db.add(metric)

    # Persist AuditTraces
    for trace in state.get("audit_trail", []):
        audit = AuditTrace(
            request_id=req_id,
            step_index=trace["step_index"],
            step_name=trace["step_name"],
            payload=trace.get("payload", {}),
            timestamp=trace.get("timestamp") or datetime.now(timezone.utc),
        )
        db.add(audit)

    await db.commit()
    return req_record
