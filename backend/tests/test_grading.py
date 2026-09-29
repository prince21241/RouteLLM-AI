"""Deterministic graders. Rubric cases stay unknown without a judge."""

from app.evaluation.dataset import load_dataset
from app.evaluation.grading import grade_case


def _case(method: str):
    dataset = load_dataset()
    return next(case for case in dataset.cases if case.method == method and case.id != "rubric-sky")


def test_exact_match_ignores_case_and_a_trailing_period() -> None:
    case = next(case for case in load_dataset().cases if case.id == "fact-paris")

    assert grade_case(case, "paris.").verdict == "pass"
    assert grade_case(case, "Lyon").verdict == "fail"


def test_numeric_answer_must_be_a_single_number() -> None:
    case = next(case for case in load_dataset().cases if case.id == "math-product")

    assert grade_case(case, "323").verdict == "pass"
    assert grade_case(case, "323 because seventeen times nineteen").verdict == "fail"


def test_json_requires_valid_object_and_fields() -> None:
    case = next(case for case in load_dataset().cases if case.id == "struct-capital")

    assert grade_case(case, '{"city": "Paris", "country": "France"}').verdict == "pass"
    missing = grade_case(case, '{"city": "Paris"}')
    invalid = grade_case(case, "Paris, France")

    assert missing.verdict == "fail"
    assert "required" in missing.reasons[0]
    assert invalid.verdict == "fail"
    assert "not valid JSON" in invalid.reasons[0]


def test_rubric_is_unknown_without_a_judge() -> None:
    case = next(case for case in load_dataset().cases if case.method == "rubric")
    result = grade_case(case, "The sky is blue because it is long and I am confident.")

    assert result.verdict == "unknown"
    assert result.score is None
