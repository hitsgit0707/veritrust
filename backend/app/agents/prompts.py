"""Prompt templates and system instructions for VeriTrust AI agents."""

from typing import Any, List, Optional
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


JUDGE_SYSTEM_PROMPT = (
    "You are the Judge Agent for VeriTrust AI, an enterprise compliance and factual verification system.\n"
    "Your responsibility is to strictly and objectively audit customer support draft answers against "
    "retrieved company knowledge-base evidence.\n\n"
    "CRITICAL EVALUATION RULES:\n"
    "1. Never trust the Maker's self-assessed confidence or claim of truth. Evaluate ONLY against the actual evidence provided.\n"
    "2. If the draft contains ANY claim directly contradicting verified evidence, flag as CONTRADICTION.\n"
    "3. If the draft invents or asserts policies, tiers, or programs explicitly negated or absent from the evidence, flag as FABRICATED_POLICY.\n"
    "4. If numbers, durations, percentages, or timeframes differ from the evidence, flag as UNSUPPORTED_NUMERICAL.\n"
    "5. If an important factual statement is made with no supporting evidence in the knowledge base, flag as MISSING_EVIDENCE or UNSUPPORTED_FACT.\n"
    "6. If the Maker cites a source that does not contain or support the claim, flag as SOURCE_MISMATCH.\n"
    "7. A draft must be APPROVED (approved=true) ONLY if it contains ZERO contradictions, ZERO fabricated policies, and ALL claims are fully supported.\n"
    "8. Assign an explainable compliance score between 0 and 100 (100 = completely grounded, 0 = entirely fabricated or severe contradiction)."
)


def build_judge_prompt(
    question: str,
    draft_answer: str,
    retrieved_chunks: List[RetrievedChunk],
    sources: Optional[List[str]] = None,
    confidence: Optional[float] = None,
) -> str:
    """Format customer question, draft answer, cited sources, and retrieved evidence for Judge evaluation."""
    evidence_blocks = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        evidence_blocks.append(
            f"--- Evidence Item {idx} ---\n"
            f"[Source: {chunk.source}]\n"
            f"Document: {chunk.filename}\n"
            f"Content:\n{chunk.text.strip()}\n"
        )
    evidence_text = "\n".join(evidence_blocks) if evidence_blocks else "[NO RETRIEVED EVIDENCE PROVIDED]"

    sources_text = ", ".join(sources) if sources else "None cited"
    confidence_text = f"{confidence:.2f}" if confidence is not None else "Not provided"

    prompt = (
        f"Customer Question:\n\"{question}\"\n\n"
        f"Maker Draft Answer to Evaluate:\n\"{draft_answer}\"\n\n"
        f"Maker Cited Sources: {sources_text}\n"
        f"Maker Self-Assessed Confidence: {confidence_text}\n\n"
        f"Retrieved Company Knowledge-Base Evidence:\n"
        f"{evidence_text}\n\n"
        f"Instructions:\n"
        f"Independently audit the Maker Draft Answer against the verified evidence.\n"
        f"Determine if the draft should be approved (approved: true/false).\n"
        f"Calculate an explainable compliance score (0-100).\n"
        f"List all identified compliance issues with issue_type, description, severity, claim_text, evidence_ref, and source."
    )
    return prompt


CORRECTION_SYSTEM_PROMPT = (
    "You are the Correction Agent for VeriTrust AI, an enterprise compliance and factual verification system.\n"
    "Your responsibility is to rewrite rejected customer support draft answers using EXCLUSIVELY facts "
    "supported by the provided knowledge-base evidence, resolving all issues identified by the Judge Agent.\n\n"
    "CRITICAL CORRECTION RULES:\n"
    "1. Read all issues flagged by the Judge Agent.\n"
    "2. Remove any unsupported claims, fabricated policies, or inaccurate statements.\n"
    "3. Correct contradictions and incorrect numerical/timeframe values to align precisely with verified evidence.\n"
    "4. Preserve accurate, grounded parts of the original draft where possible.\n"
    "5. NEVER invent replacement facts, terms, or unverified claims.\n"
    "6. If the evidence does not contain sufficient information to answer the question, state clearly that "
    "the information could not be verified from the available company knowledge base.\n"
    "7. Provide a concise explanation of what was removed or corrected."
)


def build_correction_prompt(
    question: str,
    draft_answer: str,
    issues: List[Any],
    retrieved_chunks: List[RetrievedChunk],
) -> str:
    """Format customer question, rejected draft, Judge issues, and evidence for Correction Agent."""
    evidence_blocks = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        evidence_blocks.append(
            f"--- Evidence Item {idx} ---\n"
            f"[Source: {chunk.source}]\n"
            f"Document: {chunk.filename}\n"
            f"Content:\n{chunk.text.strip()}\n"
        )
    evidence_text = "\n".join(evidence_blocks) if evidence_blocks else "[NO RETRIEVED EVIDENCE PROVIDED]"

    issue_lines = []
    for idx, issue in enumerate(issues, start=1):
        itype = getattr(issue, "issue_type", str(issue.get("issue_type") if isinstance(issue, dict) else issue))
        desc = getattr(issue, "description", str(issue.get("description") if isinstance(issue, dict) else ""))
        claim = getattr(issue, "claim_text", getattr(issue, "claim", str(issue.get("claim_text") if isinstance(issue, dict) else "")))
        issue_lines.append(f"{idx}. [{itype}] {desc} (Offending claim: '{claim}')")

    issues_text = "\n".join(issue_lines) if issue_lines else "None specified"

    prompt = (
        f"Customer Question:\n\"{question}\"\n\n"
        f"Rejected Draft Answer:\n\"{draft_answer}\"\n\n"
        f"Judge Identified Compliance Issues:\n{issues_text}\n\n"
        f"Retrieved Company Knowledge-Base Evidence:\n{evidence_text}\n\n"
        f"Instructions:\n"
        f"Rewrite the draft answer so it is 100% compliant with the verified evidence.\n"
        f"Remove or correct every identified issue.\n"
        f"Provide your 'corrected_answer' and an 'explanation' detailing what was corrected."
    )
    return prompt
