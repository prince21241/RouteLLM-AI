"""Deterministic graders for verifiable cases.

Rubric cases are not graded here. Keyword overlap, answer length, and a
model's stated confidence are not checks.
"""

import json
from decimal import Decimal, InvalidOperation

from app.evaluation.dataset import EvalCase
from app.evaluation.schema import EvaluationResult


def grade_case(case: EvalCase, answer: str) -> EvaluationResult:
    """Grade ``answer`` against the case. Rubric cases stay unknown."""
    if case.method == "exact":
        return _exact(case, answer)
    if case.method == "numeric":
        return _numeric(case, answer)
    if case.method == "json_fields":
        return _json_fields(case, answer)
    return EvaluationResult(
        verdict="unknown",
        score=None,
        reasons=("Rubric cases need a model judge. Dataset grading does not infer a pass.",),
        method="rubric",
    )


def _exact(case: EvalCase, answer: str) -> EvaluationResult:
    expected = _normalize_text(str(case.expected))
    actual = _normalize_text(answer)
    if actual == expected:
        return _pass("exact", "The answer matches the reference.")
    return _fail("exact", "The answer does not match the reference.")


def _numeric(case: EvalCase, answer: str) -> EvaluationResult:
    expected = Decimal(str(case.expected))
    try:
        actual = Decimal(answer.strip().rstrip("."))
    except InvalidOperation:
        return _fail("numeric", "The answer is not a single number.")
    if actual == expected:
        return _pass("numeric", "The number matches the reference.")
    return _fail("numeric", "The number does not match the reference.")


def _json_fields(case: EvalCase, answer: str) -> EvaluationResult:
    parsed = _load_object(answer)
    if isinstance(parsed, str):
        return _fail("json_fields", parsed)
    expected = case.expected
    if not isinstance(expected, dict):
        return EvaluationResult(
            verdict="error",
            score=None,
            reasons=("The case expected value is not an object.",),
            method="json_fields",
            error_message="invalid dataset case",
        )
    missing = [key for key in expected if key not in parsed]
    if missing:
        return _fail("json_fields", "The JSON object is missing required fields.")
    mismatched = [key for key in expected if not _values_match(parsed[key], expected[key])]
    if mismatched:
        return _fail("json_fields", "A required JSON field does not match the reference.")
    return _pass("json_fields", "The JSON object contains the required fields.")


def _load_object(answer: str) -> dict[str, object] | str:
    text = answer.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return "The answer is not valid JSON."
    if not isinstance(value, dict):
        return "The answer JSON is not an object."
    return value


def _values_match(actual: object, expected: object) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        return _normalize_text(actual) == _normalize_text(expected)
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual is expected
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return Decimal(str(actual)) == Decimal(str(expected))
    return actual == expected


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().casefold().rstrip(".").split())


def _pass(method: str, reason: str) -> EvaluationResult:
    return EvaluationResult(verdict="pass", score=1.0, reasons=(reason,), method=method)


def _fail(method: str, reason: str) -> EvaluationResult:
    return EvaluationResult(verdict="fail", score=0.0, reasons=(reason,), method=method)
