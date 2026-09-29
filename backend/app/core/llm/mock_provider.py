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
            if "case 2" in lower_prompt or "45 days" in lower_prompt:
                data = {
                    "draft_answer": "Refunds are accepted within 45 days with original receipt.",
                    "sources": ["sample_data/refund_policy.txt#chunk_1"],
                    "confidence": 0.88,
                }
            elif "case 3" in lower_prompt or "premium" in lower_prompt:
                data = {
                    "draft_answer": "Premium users receive free replacement anytime under our VIP care policy.",
                    "sources": ["sample_data/product_catalog.csv#chunk_1"],
                    "confidence": 0.85,
                }
            elif "case 4" in lower_prompt or "2 years" in lower_prompt:
                data = {
                    "draft_answer": "All devices are covered by a 2-year comprehensive hardware warranty.",
                    "sources": ["sample_data/warranty_policy.txt#chunk_1"],
                    "confidence": 0.90,
                }
            elif "case 5" in lower_prompt or "confidence 0.99" in lower_prompt:
                data = {
                    "draft_answer": "All items are delivered same-day anywhere in the country.",
                    "sources": ["sample_data/shipping_policy.txt#chunk_1"],
                    "confidence": 0.99,
                }
            else:
                data = {
                    "draft_answer": "According to our company policy, refunds are available within 30 days of purchase.",
                    "sources": ["sample_data/refund_policy.txt#chunk_1"],
                    "confidence": 0.95,
                }

        # 2. Judge Agent Output Pattern
        elif {"approved", "score", "issues"}.issubset(field_names):
            # Adversarial Case 2: Contradiction (45 days vs 30 days)
            if "45 days" in lower_prompt:
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
                        }
                    ],
                }
            # Adversarial Case 3: Fabricated Policy (Premium replacement)
            elif "premium" in lower_prompt or "vip care" in lower_prompt:
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
                        }
                    ],
                }
            # Adversarial Case 4: Numerical / Period Drift (2 years vs 1 year)
            elif "2-year" in lower_prompt or "2 years" in lower_prompt:
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
                        }
                    ],
                }
            # Adversarial Case 5: Maker claims 0.99 confidence on wrong facts
            elif "same-day" in lower_prompt:
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
                        }
                    ],
                }
            # Adversarial Case 6: Persistent ungrounded claims
            elif "unresolvable" in lower_prompt or "retry_count: 3" in lower_prompt or "attempt 3" in lower_prompt:
                data = {
                    "approved": False,
                    "score": 25,
                    "issues": [
                        {
                            "issue_type": "MISSING_EVIDENCE",
                            "description": "Sufficient evidence could not be verified from the knowledge base after multiple correction attempts.",
                            "severity": "CRITICAL",
                            "claim_text": "Unverified claim",
                            "evidence_ref": "NONE",
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
            if "45 days" in lower_prompt:
                data = {
                    "corrected_answer": "Refunds are processed within 30 days of purchase with the original receipt.",
                    "explanation": "Corrected the refund period from 45 days to the policy-verified 30 days.",
                }
            elif "premium" in lower_prompt:
                data = {
                    "corrected_answer": "Our standard warranty and refund policies apply to all customers as specified in the catalog.",
                    "explanation": "Removed unsupported claim regarding a VIP care free replacement policy.",
                }
            elif "2-year" in lower_prompt or "2 years" in lower_prompt:
                data = {
                    "corrected_answer": "All hardware products carry a 1-year limited warranty against manufacturing defects.",
                    "explanation": "Corrected warranty duration from 2 years to 1 year per company policy.",
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
