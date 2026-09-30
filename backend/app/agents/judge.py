"""Agent 2: Judge Agent.

Responsible for independently auditing and validating Maker Agent drafts against
the retrieved company knowledge-base evidence.
"""

import asyncio
from typing import List, Optional
from pydantic import Field

from backend.app.agents.maker import MakerResult
from backend.app.agents.prompts import JUDGE_SYSTEM_PROMPT, build_judge_prompt
from backend.app.core.llm import BaseLLMProvider, get_llm_provider
from backend.app.rag.vector_store import RetrievedChunk
from backend.app.schemas.compliance import IssueItem, JudgeOutput


class JudgeResult(JudgeOutput):
    """Encapsulates the Judge's evaluation of a draft against retrieved evidence."""
    question: Optional[str] = None
    draft_answer: Optional[str] = None
    retrieved_evidence: List[RetrievedChunk] = Field(default_factory=list)


class JudgeAgent:
    """Independent compliance audit agent validating Maker drafts against RAG evidence."""

    def __init__(
        self,
        llm_provider: Optional[BaseLLMProvider] = None,
    ):
        self.llm_provider = llm_provider or get_llm_provider()

    async def evaluate(
        self,
        question: str,
        draft_answer: str,
        retrieved_evidence: List[RetrievedChunk],
        sources: Optional[List[str]] = None,
        confidence: Optional[float] = None,
    ) -> JudgeResult:
        """
        Independently evaluate a draft answer against retrieved evidence.

        The Judge does NOT trust Maker self-assessed confidence, wording, or citations.
        """
        # 1. Programmatic source verification: Check for citation hallucinations / mismatches
        valid_source_ids = set()
        for chunk in retrieved_evidence:
            if chunk.source:
                valid_source_ids.add(chunk.source)
            if chunk.chunk_id:
                valid_source_ids.add(chunk.chunk_id)

        detected_issues: List[IssueItem] = []
        for src in (sources or []):
            if src not in valid_source_ids:
                detected_issues.append(
                    IssueItem(
                        issue_type="SOURCE_MISMATCH",
                        description=f"Maker cited source '{src}' which is not in the retrieved evidence.",
                        severity="HIGH",
                        claim_text=draft_answer,
                        evidence_ref="NONE",
                        source=src,
                    )
                )

        # 2. Check for missing evidence when no evidence was retrieved
        if not retrieved_evidence:
            lower_draft = draft_answer.lower()
            cautious_keywords = ["unable to answer", "could not be verified", "no information", "cannot answer"]
            if not any(ck in lower_draft for ck in cautious_keywords):
                detected_issues.append(
                    IssueItem(
                        issue_type="MISSING_EVIDENCE",
                        description="Factual claims made without supporting knowledge base evidence.",
                        severity="CRITICAL",
                        claim_text=draft_answer,
                        evidence_ref="NONE",
                        source="NONE",
                    )
                )

        # 3. Build prompt and invoke LLM Judge
        prompt = build_judge_prompt(
            question=question,
            draft_answer=draft_answer,
            retrieved_chunks=retrieved_evidence,
            sources=sources,
            confidence=confidence,
        )

        raw_output: JudgeOutput = await self.llm_provider.generate_structured(
            prompt=prompt,
            response_model=JudgeOutput,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            temperature=0.0,
        )

        # 4. Combine detected issues and deduplicate
        all_issues: List[IssueItem] = list(detected_issues)
        existing_types = {i.issue_type for i in all_issues}
        for item in raw_output.issues:
            if item.issue_type not in existing_types or item.description not in [i.description for i in all_issues]:
                all_issues.append(item)
                existing_types.add(item.issue_type)

        # 5. Strict approval determination (independent of Maker confidence)
        has_critical_or_unsupported = any(
            i.issue_type in ("CONTRADICTION", "FABRICATED_POLICY", "UNSUPPORTED_NUMERICAL", "MISSING_EVIDENCE", "UNSUPPORTED_FACT", "SOURCE_MISMATCH")
            for i in all_issues
        )
        if len(all_issues) > 0 or has_critical_or_unsupported:
            final_approved = False
        else:
            final_approved = raw_output.approved

        # 6. Explainable score calculation
        if final_approved:
            final_score = max(raw_output.score, 85)
        else:
            penalty = 0
            for iss in all_issues:
                sev = iss.severity.upper()
                if sev == "CRITICAL":
                    penalty += 40
                elif sev == "HIGH":
                    penalty += 25
                elif sev == "MEDIUM":
                    penalty += 15
                else:
                    penalty += 10
            # A rejected answer score cannot exceed 60
            calculated_score = max(0, 100 - penalty)
            final_score = min(raw_output.score, calculated_score, 60)

        return JudgeResult(
            approved=final_approved,
            score=final_score,
            issues=all_issues,
            question=question,
            draft_answer=draft_answer,
            retrieved_evidence=retrieved_evidence,
        )

    async def evaluate_maker_result(
        self,
        maker_result: MakerResult,
    ) -> JudgeResult:
        """Convenience method to evaluate a MakerResult directly."""
        return await self.evaluate(
            question=maker_result.question,
            draft_answer=maker_result.draft_answer,
            retrieved_evidence=maker_result.retrieved_evidence,
            sources=maker_result.sources,
            confidence=maker_result.confidence,
        )

    def evaluate_sync(
        self,
        question: str,
        draft_answer: str,
        retrieved_evidence: List[RetrievedChunk],
        sources: Optional[List[str]] = None,
        confidence: Optional[float] = None,
    ) -> JudgeResult:
        """Synchronous wrapper for evaluate."""
        return asyncio.run(
            self.evaluate(
                question=question,
                draft_answer=draft_answer,
                retrieved_evidence=retrieved_evidence,
                sources=sources,
                confidence=confidence,
            )
        )

    def evaluate_maker_result_sync(
        self,
        maker_result: MakerResult,
    ) -> JudgeResult:
        """Synchronous wrapper for evaluate_maker_result."""
        return asyncio.run(self.evaluate_maker_result(maker_result))
