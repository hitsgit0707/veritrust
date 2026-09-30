"""Agents package exports for VeriTrust AI."""

from backend.app.agents.judge import JudgeAgent, JudgeResult
from backend.app.agents.maker import MakerAgent, MakerResult
from backend.app.agents.prompts import (
    JUDGE_SYSTEM_PROMPT,
    MAKER_SYSTEM_PROMPT,
    build_judge_prompt,
    build_maker_prompt,
)

__all__ = [
    "MakerAgent",
    "MakerResult",
    "JudgeAgent",
    "JudgeResult",
    "MAKER_SYSTEM_PROMPT",
    "JUDGE_SYSTEM_PROMPT",
    "build_maker_prompt",
    "build_judge_prompt",
]
