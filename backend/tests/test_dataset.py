"""Dataset contract."""

import json
from pathlib import Path

import pytest

from app.evaluation.dataset import (
    CATEGORIES,
    DATASET_PATH,
    FAMILIES_PATH,
    SPLITS,
    DatasetError,
    load_dataset,
    select_cases,
)
from app.ml.schema import TrainingDataError, load_training_dataset


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


def test_smoke_dataset_is_not_training_evidence() -> None:
    dataset = load_dataset()

    assert dataset.version == "2026-09-29.1"
    assert all(case.group_id is None for case in dataset.cases)
    with pytest.raises(TrainingDataError, match="not ML training evidence"):
        load_training_dataset(DATASET_PATH)


def test_families_dataset_groups_paraphrases() -> None:
    dataset = load_dataset(FAMILIES_PATH)

    assert dataset.version == "2026-09-29.families"
    assert len(dataset.cases) == 96
    grouped: dict[str, list] = {}
    for case in dataset.cases:
        assert case.group_id is not None
        grouped.setdefault(case.group_id, []).append(case)
    assert len(grouped) == 48
    for category in CATEGORIES:
        assert sum(1 for group in grouped.values() if group[0].category == category) == 8
    for members in grouped.values():
        assert len(members) == 2
        assert len({case.prompt for case in members}) == 2
        assert len({(case.split, case.category, case.method) for case in members}) == 1
        assert len({json.dumps(case.expected, sort_keys=True) for case in members}) == 1
        assert len({case.criteria for case in members}) == 1
        if members[0].method == "rubric":
            assert members[0].criteria
        else:
            assert members[0].expected is not None
    assert {case.split for case in dataset.cases} == SPLITS
    with pytest.raises(TrainingDataError, match="not ML training evidence"):
        load_training_dataset(FAMILIES_PATH)


def test_held_out_split_does_not_include_calibration() -> None:
    dataset = load_dataset()
    held_out = select_cases(dataset, split="held_out")

    assert held_out
    assert all(case.split == "held_out" for case in held_out)
