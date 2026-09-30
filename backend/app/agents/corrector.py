"""Agent 3: Correction Agent.

Responsible for rewriting non-compliant Maker drafts to strictly adhere to verified
company knowledge-base evidence, resolving all issues identified by the Judge Agent.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union

from backend.app.agents.prompts import (
    CORRECTION_SYSTEM_PROMPT,
    build_correction_prompt,
)
from backend.app.core.llm import BaseLLMProvider, get_llm_provider
from backend.app.rag.vector_store import RetrievedChunk
from backend.app.schemas.compliance import (
    CorrectionOutput,
    CorrectionResult,
    IssueItem,
    JudgeOutput,
)


class CorrectionAgent:
    """Agent that rewrites rejected drafts to remove hallucinations and align with verified evidence."""

    def __init__(
        self,
        llm_provider: Optional[BaseLLMProvider] = None,
    ):
        self.llm_provider = llm_provider or get_llm_provider()

    async def correct(
        self,
        question: str,
        draft_answer: str,
        judge_result: Union[JudgeOutput, Dict[str, Any]],
        retrieved_evidence: List[RetrievedChunk],
    ) -> CorrectionResult:
        """
        Rewrite draft answer resolving all compliance issues.

        Enforces that the corrected answer relies ONLY on verified knowledge-base evidence.
        """
        # Extract issues from Judge result
        if isinstance(judge_result, dict):
            raw_issues = judge_result.get("issues", [])
        else:
            raw_issues = getattr(judge_result, "issues", [])

        issues: List[Any] = []
        for item in raw_issues:
            if isinstance(item, dict):
                issues.append(IssueItem.model_validate(item))
            elif isinstance(item, IssueItem):
                issues.append(item)
            else:
                issues.append(item)

        # Build prompt
        prompt = build_correction_prompt(
            question=question,
            draft_answer=draft_answer,
            issues=issues,
            retrieved_chunks=retrieved_evidence,
        )

        # Generate structured correction
        raw_output: CorrectionOutput = await self.llm_provider.generate_structured(
            prompt=prompt,
            response_model=CorrectionOutput,
            system_prompt=CORRECTION_SYSTEM_PROMPT,
            temperature=0.0,
        )

        corrections_list = [raw_output.explanation]
        for issue in issues:
            desc = getattr(issue, "description", None)
            if desc and desc not in corrections_list:
                corrections_list.append(f"Resolved [{getattr(issue, 'issue_type', 'ISSUE')}]: {desc}")

        return CorrectionResult(
            corrected_answer=raw_output.corrected_answer,
            explanation=raw_output.explanation,
            corrections=corrections_list,
        )

    def correct_sync(
        self,
        question: str,
        draft_answer: str,
        judge_result: Union[JudgeOutput, Dict[str, Any]],
        retrieved_evidence: List[RetrievedChunk],
    ) -> CorrectionResult:
        """Synchronous wrapper for correct."""
        return asyncio.run(
            self.correct(
                question=question,
                draft_answer=draft_answer,
                judge_result=judge_result,
                retrieved_evidence=retrieved_evidence,
            )
        )
