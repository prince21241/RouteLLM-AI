"""Behavioral tests for the rule-based complexity heuristic."""

import pytest

from app.api.schemas import QualityTier
from app.routing.complexity import ComplexityClassifier, tier_for_score

SIMPLE = "Explain what an API is in two sentences"
MEDIUM = (
    "Compare the tradeoffs of REST and GraphQL for a small team. "
    "Include a table of the main differences, and also list when you would choose each one."
)
COMPLEX = (
    "Debug this Python function and prove why the algorithm is incorrect. "
    "Then implement a fixed version, analyze its complexity, and return the result as JSON "
    "with a step-by-step explanation of at least 500 words."
)
LOW = 0.30
HIGH = 0.70


def classify(prompt: str, system_prompt: str | None = None, *, low: float = LOW, high: float = HIGH):
    return ComplexityClassifier().classify(prompt, system_prompt, low=low, high=high)


def test_representative_prompts_land_in_expected_bands() -> None:
    simple = classify(SIMPLE)
    medium = classify(MEDIUM)
    complex_prompt = classify(COMPLEX)

    assert simple.tier is QualityTier.LOW
    assert simple.score <= LOW
    assert "short response requested" in simple.reasons
    assert "debugging" not in simple.reasons
    assert "code generation" not in simple.reasons

    assert medium.tier is QualityTier.MEDIUM
    assert LOW < medium.score <= HIGH
    assert "analysis request" in medium.reasons
    assert "structured output" in medium.reasons
    assert "multiple instructions" in medium.reasons

    assert complex_prompt.tier is QualityTier.HIGH
    assert complex_prompt.score > HIGH
    assert "debugging" in complex_prompt.reasons
    assert "code generation" in complex_prompt.reasons
    assert "mathematical reasoning" in complex_prompt.reasons
    assert "long response requested" in complex_prompt.reasons


def test_scores_are_bounded_and_deterministic() -> None:
    prompts = [SIMPLE, MEDIUM, COMPLEX, "Hi", "What is 2 + 2?"]
    for prompt in prompts:
        first = classify(prompt)
        second = classify(prompt)
        assert first == second
        assert 0.0 <= first.score <= 1.0
        assert first.reasons


def test_threshold_boundaries_are_exact() -> None:
    assert tier_for_score(0.0, low=LOW, high=HIGH) is QualityTier.LOW
    assert tier_for_score(0.30, low=LOW, high=HIGH) is QualityTier.LOW
    assert tier_for_score(0.31, low=LOW, high=HIGH) is QualityTier.MEDIUM
    assert tier_for_score(0.70, low=LOW, high=HIGH) is QualityTier.MEDIUM
    assert tier_for_score(0.71, low=LOW, high=HIGH) is QualityTier.HIGH
    assert tier_for_score(1.0, low=LOW, high=HIGH) is QualityTier.HIGH


def test_classifier_uses_configured_thresholds() -> None:
    assessment = classify("Hi", low=0.0, high=0.04)

    assert assessment.score > 0.04
    assert assessment.tier is QualityTier.HIGH


def test_system_instructions_change_the_score() -> None:
    alone = classify("Hello")
    with_system = classify(
        "Hello",
        "Debug this Python function and return JSON.",
    )

    assert with_system.score > alone.score
    assert "debugging" in with_system.reasons
    assert "code generation" in with_system.reasons
    assert "structured output" in with_system.reasons
    assert alone.tier is QualityTier.LOW
    assert with_system.tier is not QualityTier.LOW


@pytest.mark.parametrize(
    "prompt",
    [SIMPLE, MEDIUM, COMPLEX, "Hello\n\nDebug this Python function and return JSON."],
)
def test_same_prompt_does_not_depend_on_call_order(prompt: str) -> None:
    classifier = ComplexityClassifier()
    assert classifier.classify(prompt, low=LOW, high=HIGH) == classifier.classify(
        prompt, low=LOW, high=HIGH
    )
