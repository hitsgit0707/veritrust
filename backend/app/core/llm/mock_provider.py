"""Deterministic Mock LLM Provider for offline testing, demos, and CI."""

import json
import re
from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel
from backend.app.core.llm.base import BaseLLMProvider

T = TypeVar("T", bound=BaseModel)


class MockLLMProvider(BaseLLMProvider):
    """Deterministic mock LLM provider requiring no external API keys or network access."""

    def __init__(self, default_response: Optional[str] = None):
        self._default_text = default_response or "This is a deterministic mock response from VeriTrust AI."
        self._custom_structured_response: Optional[BaseModel] = None
        self._custom_text_response: Optional[str] = None
        self._call_history: List[Dict[str, Any]] = []

    def set_mock_response(self, text: str) -> None:
        """Set an explicit text response for subsequent calls."""
        self._custom_text_response = text

    def set_structured_response(self, model_instance: BaseModel) -> None:
        """Set an explicit structured response for subsequent calls."""
        self._custom_structured_response = model_instance

    def clear_overrides(self) -> None:
        """Clear any explicit overrides and reset call history."""
        self._custom_text_response = None
        self._custom_structured_response = None
        self._call_history.clear()

    @property
    def call_history(self) -> List[Dict[str, Any]]:
        """Return history of all calls made to this provider."""
        return self._call_history

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        """Generate deterministic text response."""
        self._call_history.append({
            "type": "text",
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
        })

        if self._custom_text_response is not None:
            return self._custom_text_response

        lower_prompt = prompt.lower()
        if "refund" in lower_prompt:
            if "45 days" in lower_prompt:
                return "Refunds are processed within 30 days of purchase per policy, not 45 days."
            return "Refunds are available within 30 days of purchase with original receipt."
        if "warranty" in lower_prompt:
            return "Our hardware products include a 1-year limited warranty."
        if "shipping" in lower_prompt:
            return "Standard shipping takes 3-5 business days."

        return self._default_text

    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> T:
        """Generate deterministic structured response matching the requested Pydantic model."""
        self._call_history.append({
            "type": "structured",
            "prompt": prompt,
            "response_model": response_model.__name__,
            "system_prompt": system_prompt,
            "temperature": temperature,
        })

        if self._custom_structured_response is not None and isinstance(self._custom_structured_response, response_model):
            return self._custom_structured_response

        # Synthesize domain-aware response based on prompt cues and model fields
        field_names = set(response_model.model_fields.keys())
        data: Dict[str, Any] = {}
        lower_prompt = prompt.lower()

        # 1. Maker Agent Output Pattern
        if {"draft_answer", "sources", "confidence"}.issubset(field_names):
            found_sources = re.findall(r"\[Source:\s*([^\]]+)\]", prompt)

            # Isolate query directives from retrieved knowledge chunks so evidence text doesn't trigger false positives
            query_part = prompt.split("Retrieved Company Knowledge-Base Evidence:")[0].lower()

            # Extract specific customer question and adversarial mode if present
            question_match = re.search(r'Customer Question:\s*\n?"?([^"\n]+)"?', prompt, re.IGNORECASE)
            c_q = question_match.group(1).lower().strip() if question_match else query_part
            adv_match = re.search(r'ADVERSARIAL_MODE:\s*([^\n]+)', prompt, re.IGNORECASE)
            adv_mode = adv_match.group(1).lower().strip() if adv_match else ""

            # Persistent failure scenario: always produces an unverifiable claim so the Judge
            # rejects it every time, causing the workflow to exhaust 3 retries and BLOCK.
            if "persistent_failure" in adv_mode:
                data = {
                    "draft_answer": "We accept bitcoin and dogecoin payments with zero fees as a premium payment option.",
                    "sources": [],
                    "confidence": 0.90,
                }
            # Insufficient evidence scenario
            elif "no relevant evidence" in lower_prompt or "insufficient" in query_part or any(k in c_q for k in ["cryptocurrency", "dogecoin", "bitcoin", "gold bullion", "pet dog", "out-of-scope"]) or (not found_sources and "case" not in query_part and "45 days" not in c_q and "warranty" not in c_q and "shipping" not in c_q and "refund" not in c_q):
                data = {
                    "draft_answer": "I am unable to answer your question because the required information could not be verified from our available company knowledge base.",
                    "sources": [],
                    "confidence": 0.20,
                }
            # Adversarial Case 2: Contradiction (45 days)
            elif "case 2" in adv_mode or "case 2" in query_part or "45 days" in query_part:
                data = {
                    "draft_answer": "Refunds are accepted within 45 days with original receipt.",
                    "sources": [found_sources[0]] if found_sources else ["refund_policy.txt#chunk_0"],
                    "confidence": 0.88,
                }
            # Adversarial Case 3: Fabricated Policy (Premium replacement)
            elif "case 3" in adv_mode or "case 3" in query_part or "vip care" in query_part or "vip" in c_q or "premium" in c_q:
                data = {
                    "draft_answer": "Premium users receive free replacement anytime under our VIP care policy.",
                    "sources": [found_sources[0]] if found_sources else ["warranty_policy.txt#chunk_0"],
                    "confidence": 0.85,
                }
            # Adversarial Case 4: Numerical / Period Drift (2 years vs 1 year)
            elif "case 4" in adv_mode or "case 4" in query_part or "2 years" in query_part or "2-year" in query_part:
                data = {
                    "draft_answer": "All devices are covered by a 2-year comprehensive hardware warranty.",
                    "sources": [found_sources[0]] if found_sources else ["warranty_policy.txt#chunk_0"],
                    "confidence": 0.90,
                }
            # Adversarial Case 5: Overconfident wrong facts
            elif "case 5" in adv_mode or "case 5" in query_part or "same-day" in query_part or "confidence 0.99" in query_part:
                data = {
                    "draft_answer": "All items are delivered same-day anywhere in the country.",
                    "sources": [found_sources[0]] if found_sources else ["shipping_policy.txt#chunk_0"],
                    "confidence": 0.99,
                }
            # Grounded Domain Queries
            elif "warranty" in query_part:
                data = {
                    "draft_answer": "According to our company warranty policy, all hardware products carry a standard 1-year limited warranty from the confirmed date of purchase.",
                    "sources": [found_sources[0]] if found_sources else ["warranty_policy.txt#chunk_0"],
                    "confidence": 0.95,
                }
            elif "shipping" in query_part:
                data = {
                    "draft_answer": "Standard domestic shipping takes 3 to 5 business days and is free on orders over $50.",
                    "sources": [found_sources[0]] if found_sources else ["shipping_policy.txt#chunk_0"],
                    "confidence": 0.95,
                }
            elif "headset" in query_part or "ultrasound" in query_part or "catalog" in query_part or "prod-101" in query_part:
                data = {
                    "draft_answer": "The UltraSound Pro Wireless Headset (SKU: PROD-101) is priced at $99.99 and includes a 12-month warranty.",
                    "sources": [found_sources[0]] if found_sources else ["product_catalog.csv#chunk_0"],
                    "confidence": 0.95,
                }
            else:
                data = {
                    "draft_answer": "According to our company policy, refunds are available within 30 days of purchase with the original receipt.",
                    "sources": [found_sources[0]] if found_sources else ["refund_policy.txt#chunk_0"],
                    "confidence": 0.95,
                }

        # 2. Judge Agent Output Pattern
        elif {"approved", "score", "issues"}.issubset(field_names):
            # Isolate draft and question from retrieved evidence to avoid keywords in evidence triggering false rejections
            eval_part = prompt.split("Retrieved Company Knowledge-Base Evidence:")[0].lower()

            # For adversarial keyword routing, extract only the draft answer section.
            # The Customer Question line persists across all iterations (including after correction),
            # so routing on the full eval_part would cause corrected answers to keep being rejected.
            if "maker draft answer to evaluate:" in eval_part:
                draft_section = eval_part.split("maker draft answer to evaluate:")[1]
                if "maker cited sources:" in draft_section:
                    draft_section = draft_section.split("maker cited sources:")[0]
            else:
                draft_section = eval_part

            # Source Mismatch scenario — check full eval_part since source info is in metadata
            if "source_mismatch" in eval_part or "fabricated_source" in eval_part or "fake_source" in eval_part or "source mismatch" in eval_part:

                data = {
                    "approved": False,
                    "score": 40,
                    "issues": [
                        {
                            "issue_type": "SOURCE_MISMATCH",
                            "description": "Maker cited a source that does not exist in the retrieved evidence or does not support the draft claim.",
                            "severity": "HIGH",
                            "claim_text": "Claim citing invalid source",
                            "evidence_ref": "NONE",
                            "source": "fabricated_source.txt#chunk_99",
                        }
                    ],
                }
            # Adversarial Case 2: Contradiction (45 days vs 30 days)
            elif "45 days" in draft_section:
                data = {
                    "approved": False,
                    "score": 40,
                    "issues": [
                        {
                            "issue_type": "CONTRADICTION",
                            "description": "Draft claims a 45-day refund window, but verified policy specifies 30 days.",
                            "severity": "CRITICAL",
                            "claim_text": "Refunds are accepted within 45 days with original receipt.",
                            "evidence_ref": "Refunds are available within 30 days of purchase.",
                            "source": "refund_policy.txt#chunk_0",
                        }
                    ],
                }
            # Adversarial Case 3: Fabricated Policy (Premium replacement)
            elif "premium" in draft_section or "vip care" in draft_section or "vip" in draft_section:
                data = {
                    "approved": False,
                    "score": 30,
                    "issues": [
                        {
                            "issue_type": "FABRICATED_POLICY",
                            "description": "Draft claims a free premium replacement policy not found anywhere in company records.",
                            "severity": "CRITICAL",
                            "claim_text": "Premium users receive free replacement anytime under our VIP care policy.",
                            "evidence_ref": "NONE",
                            "source": "warranty_policy.txt#chunk_0",
                        }
                    ],
                }
            # Adversarial Case 4: Numerical / Period Drift (2 years vs 1 year)
            elif "2-year" in draft_section or "2 years" in draft_section:
                data = {
                    "approved": False,
                    "score": 45,
                    "issues": [
                        {
                            "issue_type": "UNSUPPORTED_NUMERICAL",
                            "description": "Warranty period stated as 2 years; verified policy states 1 year limited hardware warranty.",
                            "severity": "HIGH",
                            "claim_text": "covered by a 2-year comprehensive hardware warranty.",
                            "evidence_ref": "All hardware products carry a 1-year limited warranty.",
                            "source": "warranty_policy.txt#chunk_0",
                        }
                    ],
                }
            # Adversarial Case 5: Maker claims 0.99 confidence on wrong facts
            elif "same-day" in draft_section:
                data = {
                    "approved": False,
                    "score": 35,
                    "issues": [
                        {
                            "issue_type": "CONTRADICTION",
                            "description": "Draft claims same-day delivery; verified shipping policy specifies 3-5 business days.",
                            "severity": "CRITICAL",
                            "claim_text": "All items are delivered same-day anywhere in the country.",
                            "evidence_ref": "Standard shipping takes 3-5 business days.",
                            "source": "shipping_policy.txt#chunk_0",
                        }
                    ],
                }
            # Adversarial Case 6: Persistent ungrounded claims / Missing Evidence / Out of Scope
            # Check draft_section for bitcoin/dogecoin (persistent_failure Maker output) and
            # eval_part for "unresolvable" (persistent_failure Correction output).
            elif any(k in draft_section for k in ["bitcoin", "dogecoin", "cryptocurrency", "gold bullion", "out-of-scope"]) or \
                 any(k in eval_part for k in ["unresolvable", "persistent_failure", "retry_count: 3", "attempt 3", "missing evidence"]):
                data = {
                    "approved": False,
                    "score": 25,
                    "issues": [
                        {
                            "issue_type": "MISSING_EVIDENCE",
                            "description": "Sufficient evidence could not be verified from the knowledge base to support the factual claims in the draft.",
                            "severity": "CRITICAL",
                            "claim_text": "Payment accepted in gold bullion or bitcoin",
                            "evidence_ref": "NONE",
                            "source": "NONE",
                        }
                    ],
                }
            # Case 1 / Verified answer: Approval
            else:
                data = {
                    "approved": True,
                    "score": 95,
                    "issues": [],
                }


        # 3. Correction Agent Output Pattern
        elif {"corrected_answer", "explanation"}.issubset(field_names):
            if any(k in lower_prompt for k in ["persistent_failure", "unresolvable", "fail_correction", "bitcoin", "dogecoin", "cryptocurrency"]):
                data = {
                    "corrected_answer": "Unresolvable claim: We cannot verify this payment method against the company knowledge base.",
                    "explanation": "Unable to verify this claim against available company records after multiple attempts.",
                }
            elif "45 days" in lower_prompt:
                data = {
                    "corrected_answer": "Refunds are processed within 30 days of purchase with the original receipt in accordance with company policy.",
                    "explanation": "Corrected the refund period from 45 days to the policy-verified 30 days.",
                }
            elif "premium" in lower_prompt or "vip" in lower_prompt:
                data = {
                    "corrected_answer": "According to our company warranty policy, all hardware products carry a standard 1-year limited warranty from the confirmed date of purchase. No special replacement tiers exist.",
                    "explanation": "Removed unsupported claim regarding a free replacement policy.",
                }
            elif "2-year" in lower_prompt or "2 years" in lower_prompt:
                data = {
                    "corrected_answer": "All hardware products carry a standard 1-year limited warranty from the confirmed date of purchase.",
                    "explanation": "Corrected warranty duration from 2 years to 1 year per company policy.",
                }
            elif "same-day" in lower_prompt:
                data = {
                    "corrected_answer": "Standard domestic shipping takes 3 to 5 business days and is free on orders over $50.",
                    "explanation": "Corrected delivery timeframe from same-day delivery to standard shipping of 3-5 business days.",
                }
            else:
                data = {
                    "corrected_answer": "Refunds are available within 30 days of purchase in accordance with company policy.",
                    "explanation": "Aligned draft precisely with verified knowledge base evidence.",
                }

        # Fallback: Populate field defaults based on types
        if not data:
            for field, field_info in response_model.model_fields.items():
                annotation = field_info.annotation
                if annotation in (str, Optional[str]):
                    data[field] = "Mock generated value"
                elif annotation in (int, Optional[int]):
                    data[field] = 90
                elif annotation in (float, Optional[float]):
                    data[field] = 0.95
                elif annotation in (bool, Optional[bool]):
                    data[field] = True
                elif getattr(annotation, "__origin__", None) is list:
                    data[field] = []
                elif getattr(annotation, "__origin__", None) is dict:
                    data[field] = {}
                else:
                    data[field] = None

        return response_model.model_validate(data)
