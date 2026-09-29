"""Typed evaluation results.

Dataset grading may use a reference answer. Live grading does not.
A verdict of ``unknown`` or ``error`` is not a pass and is not a failure
that authorizes escalation. Only ``fail`` is an explicit quality failure.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from app.api.schemas import ReportedUsage
from app.pricing.service import CostEstimate

Verdict = Literal["pass", "fail", "unknown", "error"]


@dataclass(frozen=True)
class EvaluationResult:
    """One quality check.

    ``score`` is set when the check has a numeric result. It is not inferred
    from answer length, keyword overlap, or a model's stated confidence.
    """

    verdict: Verdict
    score: float | None
    reasons: tuple[str, ...]
    method: str
    judge_model: str | None = None
    judge_usage: ReportedUsage | None = None
    judge_latency_ms: float | None = None
    judge_cost: CostEstimate | None = None
    error_message: str | None = None

    def score_decimal(self) -> Decimal | None:
        if self.score is None:
            return None
        return Decimal(str(self.score))
