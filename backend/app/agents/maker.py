"""Agent 1: Maker Agent.

Responsible for receiving customer inquiries, retrieving grounded company evidence,
and formulating draft responses citing verified knowledge-base chunks.
"""

import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field

from backend.app.agents.prompts import MAKER_SYSTEM_PROMPT, build_maker_prompt
from backend.app.core.llm import BaseLLMProvider, get_llm_provider
from backend.app.rag import RAGService, RetrievedChunk, get_rag_service
from backend.app.schemas.compliance import MakerOutput


class MakerResult(BaseModel):
    """Encapsulates the Maker's draft along with the underlying retrieved evidence."""
    question: str
    draft_answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    retrieved_evidence: List[RetrievedChunk] = Field(default_factory=list)


class MakerAgent:
    """Conversational drafting agent grounded in RAG evidence."""

    def __init__(
        self,
        llm_provider: Optional[BaseLLMProvider] = None,
        rag_service: Optional[RAGService] = None,
    ):
        self.llm_provider = llm_provider or get_llm_provider()
        self.rag_service = rag_service or get_rag_service()

    async def generate_draft(
        self,
        question: str,
        top_k: int = 4,
        adversarial_mode: Optional[str] = None,
    ) -> MakerResult:
        """
        Execute RAG retrieval and generate a grounded draft answer.
        
        Preserves the exact evidence chunks so the Judge Agent in Stage 5
        can perform strict factual verification.
        """
        # 1. Retrieve knowledge-base evidence
        chunks: List[RetrievedChunk] = self.rag_service.retrieve(query=question, top_k=top_k)

        # 2. Build structured prompt for LLM
        prompt = build_maker_prompt(
            question=question,
            retrieved_chunks=chunks,
            adversarial_mode=adversarial_mode,
        )

        # 3. Request structured response from LLM provider
        raw_output: MakerOutput = await self.llm_provider.generate_structured(
            prompt=prompt,
            response_model=MakerOutput,
            system_prompt=MAKER_SYSTEM_PROMPT,
            temperature=0.0,
        )

        # 4. Strict source validation: verify cited sources exist in retrieved evidence
        valid_source_ids = set()
        for c in chunks:
            valid_source_ids.add(c.source)
            valid_source_ids.add(c.chunk_id)

        validated_sources = [s for s in raw_output.sources if s in valid_source_ids]

        # Clamp confidence to [0.0, 1.0]
        clamped_confidence = max(0.0, min(1.0, float(raw_output.confidence)))

        return MakerResult(
            question=question,
            draft_answer=raw_output.draft_answer,
            sources=validated_sources,
            confidence=round(clamped_confidence, 2),
            retrieved_evidence=chunks,
        )

    def generate_draft_sync(
        self,
        question: str,
        top_k: int = 4,
        adversarial_mode: Optional[str] = None,
    ) -> MakerResult:
        """Synchronous wrapper for generate_draft."""
        return asyncio.run(
            self.generate_draft(
                question=question,
                top_k=top_k,
                adversarial_mode=adversarial_mode,
            )
        )
