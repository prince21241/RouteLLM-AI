"""Fit a TF-IDF logistic regression router on training prompts only.

Predicted probabilities are confidence estimates for the selected class.
They are not measurements of answer quality.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ml.split import SplitAssignment

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass(frozen=True)
class TrainedRouter:
    """A fitted pipeline plus the validation-chosen confidence threshold."""

    pipeline: object
    threshold: float
    classes: tuple[str, ...]
    validation_metrics: dict[str, object]
    exploratory: bool


def fit_router(assignment: SplitAssignment, *, seed: int) -> TrainedRouter:
    """Fit preprocessing on training groups. The test split is not read."""
    train = _rows(assignment, "train")
    validation = _rows(assignment, "validation")
    if not train or not validation:
        raise ValueError("Training and validation splits must both contain prompts")
    train_labels = {row.label for row in train}
    all_labels = {row.label for row in assignment.prompts}
    missing = sorted(all_labels - train_labels)
    if missing:
        raise ValueError(
            "Training split is missing label classes: "
            + ", ".join(missing)
            + ". Refusing to fit a classifier that cannot represent them."
        )
    pipeline = _pipeline(seed)
    pipeline.fit([row.text for row in train], [row.label for row in train])
    threshold, validation_metrics = _select_threshold(pipeline, validation)
    validation_metrics["exploratory"] = assignment.exploratory
    if assignment.exploratory_reason:
        validation_metrics["exploratory_reason"] = assignment.exploratory_reason
    classes = tuple(str(label) for label in pipeline.named_steps["clf"].classes_)
    return TrainedRouter(
        pipeline=pipeline,
        threshold=threshold,
        classes=classes,
        validation_metrics=validation_metrics,
        exploratory=assignment.exploratory,
    )


def _pipeline(seed: int) -> object:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            (
                "clf",
                LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs"),
            ),
        ]
    )


def _rows(assignment: SplitAssignment, split: str) -> list:
    return [
        prompt
        for prompt in assignment.prompts
        if assignment.group_splits[prompt.group_id] == split
    ]


def _select_threshold(pipeline: object, validation: list) -> tuple[float, dict[str, object]]:
    from sklearn.metrics import confusion_matrix

    probabilities = pipeline.predict_proba([row.text for row in validation])
    classes = [str(label) for label in pipeline.named_steps["clf"].classes_]
    truths = [row.label for row in validation]
    best_threshold = THRESHOLDS[0]
    best_key: tuple[float, int, float] | None = None
    grid = []
    for threshold in THRESHOLDS:
        covered = 0
        correct = 0
        for row, truth in zip(probabilities, truths, strict=True):
            confidence = float(row.max())
            if confidence < threshold:
                continue
            covered += 1
            predicted = classes[int(row.argmax())]
            correct += int(predicted == truth)
        accuracy = None if covered == 0 else correct / covered
        grid.append(
            {
                "threshold": threshold,
                "coverage": covered,
                "accuracy": accuracy,
            }
        )
        if covered == 0 or accuracy is None:
            continue
        key = (accuracy, covered, -threshold)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
    predicted = []
    for row in probabilities:
        predicted.append(classes[int(row.argmax())])
    matrix = confusion_matrix(truths, predicted, labels=classes).tolist()
    return best_threshold, {
        "threshold_grid": grid,
        "selected_threshold": best_threshold,
        "validation_count": len(validation),
        "validation_classes": classes,
        "validation_confusion_matrix": matrix,
        "note": (
            "Probabilities are class-confidence estimates from logistic "
            "regression. They do not measure answer quality."
        ),
    }
