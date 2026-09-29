"""Dataset contract."""

import json
from pathlib import Path

import pytest

from app.evaluation.dataset import CATEGORIES, SPLITS, DatasetError, load_dataset, select_cases


def test_shipped_dataset_covers_required_cases() -> None:
    dataset = load_dataset()

    assert dataset.version == "2026-09-29.1"
    assert len(dataset.cases) >= 25
    assert {case.split for case in dataset.cases} == SPLITS
    assert {case.category for case in dataset.cases} == CATEGORIES
    assert len(select_cases(dataset, smoke=True)) >= 3
    assert any(case.method == "rubric" and case.criteria for case in dataset.cases)
    assert any(case.method == "json_fields" for case in dataset.cases)


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    dataset = load_dataset()
    payload = dataset.model_dump()
    payload["cases"][1]["id"] = payload["cases"][0]["id"]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DatasetError):
        load_dataset(path)


def test_held_out_split_does_not_include_calibration() -> None:
    dataset = load_dataset()
    held_out = select_cases(dataset, split="held_out")

    assert held_out
    assert all(case.split == "held_out" for case in held_out)
