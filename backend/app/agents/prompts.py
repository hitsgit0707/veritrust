"""Prompt templates and system instructions for VeriTrust AI agents."""

from typing import List, Optional
from backend.app.rag.vector_store import RetrievedChunk

MAKER_SYSTEM_PROMPT = (
    "You are the Maker Agent for VeriTrust AI, an enterprise customer support assistant.\n"
    "Your responsibility is to generate accurate, professional, and helpful draft responses "
    "grounded EXCLUSIVELY in the provided company knowledge-base evidence.\n\n"
    "CRITICAL GROUNDING AND SAFETY RULES:\n"
    "1. Treat the provided retrieved evidence as the absolute factual authority.\n"
    "2. Do NOT fabricate, invent, or extrapolate policies, timeframes, warranty durations, or prices.\n"
    "3. Do NOT incorporate unsupported general or world knowledge not verified in the evidence.\n"
    "4. If the retrieved evidence does not contain the answer, you MUST state that the information "
    "could not be verified from the available company knowledge base.\n"
    "5. In the 'sources' field, cite ONLY the exact source identifiers provided in the evidence headers "
    "(e.g., 'warranty_policy.txt#chunk_0'). Never fabricate source IDs.\n"
    "6. Provide a self-assessed confidence score (0.0 to 1.0) indicating how well the evidence supports your draft.\n"
    "7. Note: Your confidence rating represents your own assessment and does NOT guarantee final approval."
)


def build_maker_prompt(
    question: str,
    retrieved_chunks: List[RetrievedChunk],
    adversarial_mode: Optional[str] = None,
) -> str:
    """Format customer question, retrieved knowledge chunks, and optional adversarial flags for Maker."""
    evidence_blocks = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        evidence_blocks.append(
            f"--- Evidence Item {idx} ---\n"
            f"[Source: {chunk.source}]\n"
            f"Document: {chunk.filename} (Similarity: {chunk.similarity_score})\n"
            f"Content:\n{chunk.text.strip()}\n"
        )

    evidence_text = "\n".join(evidence_blocks) if evidence_blocks else "[NO RELEVANT EVIDENCE FOUND IN KNOWLEDGE BASE]"

    prompt = (
        f"Customer Question:\n\"{question}\"\n\n"
        f"Retrieved Company Knowledge-Base Evidence:\n"
        f"{evidence_text}\n\n"
        f"Instructions:\n"
        f"Generate a grounded draft response addressing the customer's question. "
        f"Cite the source IDs used. Provide your confidence score between 0.0 and 1.0."
    )

    if adversarial_mode:
        prompt = f"ADVERSARIAL_MODE: {adversarial_mode}\n\n" + prompt

    return prompt
