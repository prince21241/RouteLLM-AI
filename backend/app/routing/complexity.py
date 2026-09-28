"""Rule-based prompt complexity.

The score is a bounded heuristic for routing. It does not measure true
difficulty, correctness, or how well a model will answer. Patterns are
English keyword and shape checks. They miss paraphrases and can flag
incidental words. Length is characters, not tokens.

Weights live in ``ComplexityWeights``. Change routing behavior there.
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import re

from app.api.schemas import QualityTier

_TASK_VERBS = re.compile(
    r"\b(implement|debug|analyze|analyse|prove|compare|explain|write|list|fix|calculate|design|summarize)\b",
    re.IGNORECASE,
)
_CODE = re.compile(
    r"```|\b(python|javascript|typescript|function|implement|algorithm|snippet|refactor)\b|\bclass\b",
    re.IGNORECASE,
)
_DEBUG = re.compile(
    r"\b(debug|debugging|bug|traceback|stack trace|exception)\b|\bfix\b",
    re.IGNORECASE,
)
_MATH = re.compile(
    r"\b(prove|proof|equation|derivative|integral|calculate|theorem|algebra)\b|\d+\s*[\+\-\*/]\s*\d+",
    re.IGNORECASE,
)
_ANALYSIS = re.compile(
    r"\b(analyze|analyse|analysis|compare|comparison|evaluate|evaluation|trade-?offs?)\b",
    re.IGNORECASE,
)
_STRUCTURED = re.compile(
    r"\b(json|yaml|csv|schema)\b|\btable\b|\bbullet\b",
    re.IGNORECASE,
)
_SHORT_RESPONSE = re.compile(
    r"\b(one sentence|two sentences|a sentence|briefly|in brief)\b",
    re.IGNORECASE,
)
_LONG_RESPONSE = re.compile(
    r"\b(in detail|detailed|comprehensive|step[- ]by[- ]step|in depth|essay)\b|\b\d{3,}\s+words\b",
    re.IGNORECASE,
)
_BULLETS = re.compile(r"(?m)^\s*(?:\d+[\.\)]|[-*])\s+\S")
_ALSO = re.compile(r"\band also\b|\band then\b", re.IGNORECASE)


@dataclass(frozen=True)
class ComplexityWeights:
    """Point weights on a 0-100 scale. The score is points / 100."""

    short_length: int = 5
    moderate_length: int = 15
    long_length: int = 30
    short_length_below: int = 280
    long_length_at: int = 1000
    multiple_tasks: int = 20
    code_generation: int = 20
    debugging: int = 20
    math: int = 20
    analysis: int = 15
    structured_output: int = 10
    long_response: int = 15
    short_response: int = -5


@dataclass(frozen=True)
class ComplexityAssessment:
    """Heuristic band for one request."""

    score: float
    tier: QualityTier
    reasons: tuple[str, ...]


class ComplexityClassifier:
    """Score user and system text with the shared weights."""

    def __init__(self, weights: ComplexityWeights | None = None) -> None:
        self._weights = weights or ComplexityWeights()

    def classify(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        low: float,
        high: float,
    ) -> ComplexityAssessment:
        """Return a deterministic score, tier, and the signals that fired.

        ``low`` and ``high`` are the configured thresholds. A score equal to
        a threshold stays in the lower band.
        """
        text = _instruction_text(prompt, system_prompt)
        weights = self._weights
        points = 0
        reasons: list[str] = []

        length = len(text)
        if length >= weights.long_length_at:
            points += weights.long_length
            reasons.append("long prompt")
        elif length >= weights.short_length_below:
            points += weights.moderate_length
            reasons.append("moderate prompt length")
        else:
            points += weights.short_length
            reasons.append("short prompt")

        if _multiple_tasks(text):
            points += weights.multiple_tasks
            reasons.append("multiple instructions")
        if _CODE.search(text):
            points += weights.code_generation
            reasons.append("code generation")
        if _DEBUG.search(text):
            points += weights.debugging
            reasons.append("debugging")
        if _MATH.search(text):
            points += weights.math
            reasons.append("mathematical reasoning")
        if _ANALYSIS.search(text):
            points += weights.analysis
            reasons.append("analysis request")
        if _STRUCTURED.search(text):
            points += weights.structured_output
            reasons.append("structured output")
        if _SHORT_RESPONSE.search(text):
            points += weights.short_response
            reasons.append("short response requested")
        if _LONG_RESPONSE.search(text):
            points += weights.long_response
            reasons.append("long response requested")

        bounded = min(100, max(0, points))
        score = bounded / 100
        return ComplexityAssessment(
            score=score,
            tier=tier_for_points(bounded, low=low, high=high),
            reasons=tuple(reasons),
        )


def tier_for_score(score: float, *, low: float, high: float) -> QualityTier:
    """Map a score to a tier using exact configured boundaries."""
    return tier_for_points(_units(score), low=low, high=high)


def tier_for_points(points: int, *, low: float, high: float) -> QualityTier:
    """Map hundredths of a score. ``points <= threshold`` keeps the lower tier."""
    if points <= _units(low):
        return QualityTier.LOW
    if points <= _units(high):
        return QualityTier.MEDIUM
    return QualityTier.HIGH


def _units(value: float) -> int:
    return int(
        (Decimal(str(value)) * Decimal("100")).quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP,
        )
    )


def _instruction_text(prompt: str, system_prompt: str | None) -> str:
    parts = [prompt.strip()]
    if system_prompt is not None and system_prompt.strip():
        parts.append(system_prompt.strip())
    return "\n".join(parts)


def _multiple_tasks(text: str) -> bool:
    if len(_BULLETS.findall(text)) >= 2:
        return True
    if len(_TASK_VERBS.findall(text)) >= 2:
        return True
    if _ALSO.search(text):
        return True
    return text.count("?") >= 2
