"""Local model artifacts.

Only a path inside the configured trusted directory is loaded. API requests
cannot supply a pickle or joblib path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ARTIFACT_FORMAT = "routellm-ml-1"


class ArtifactError(ValueError):
    """Raised when a local artifact is missing or incompatible."""


@dataclass(frozen=True)
class Prediction:
    """One class prediction. ``confidence`` is not an answer-quality score."""

    label: str
    confidence: float


@dataclass(frozen=True)
class LoadedArtifact:
    """A validated pipeline kept for the life of the process."""

    version: str
    threshold: float
    classes: tuple[str, ...]
    supported_model_ids: tuple[str, ...]
    pipeline: object
    metadata: dict[str, object]

    def predict(self, text: str) -> Prediction:
        probabilities = self.pipeline.predict_proba([text])[0]
        classes = [str(label) for label in self.pipeline.named_steps["clf"].classes_]
        index = int(probabilities.argmax())
        return Prediction(label=classes[index], confidence=float(probabilities[index]))


def save_artifact(
    directory: Path,
    *,
    trained: object,
    dataset_hash: str,
    split: dict[str, object],
    seed: int,
    supported_model_ids: tuple[str, ...],
    evidence: str,
    provenance: str,
) -> Path:
    """Write ``model.joblib`` and ``model.metadata.json`` in ``directory``."""
    import joblib
    import numpy
    import sklearn

    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "model.joblib"
    metadata = {
        "format": ARTIFACT_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_hash": dataset_hash,
        "split": split,
        "seed": seed,
        "supported_model_ids": list(supported_model_ids),
        "classes": list(trained.classes),
        "confidence_threshold": trained.threshold,
        "validation_metrics": trained.validation_metrics,
        "exploratory": trained.exploratory,
        "evidence": evidence,
        "provenance": provenance,
        "features": "tfidf-prompt-text",
        "estimator": "TfidfVectorizer+LogisticRegression",
        "library_versions": {
            "scikit-learn": sklearn.__version__,
            "numpy": numpy.__version__,
        },
        "probability_note": (
            "Predicted probabilities are model confidence estimates, "
            "not proof of answer quality."
        ),
    }
    joblib.dump(trained.pipeline, model_path)
    metadata_path = directory / "model.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return model_path


def load_trusted_artifact(path: Path, *, trusted_root: Path) -> LoadedArtifact:
    """Load one configured artifact after checking the sibling metadata."""
    model_path = _trusted_file(path, trusted_root)
    metadata_path = model_path.with_name("model.metadata.json")
    if model_path.name != "model.joblib":
        raise ArtifactError("Trusted artifacts must be named model.joblib")
    if not metadata_path.is_file():
        raise ArtifactError("Artifact metadata is missing")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Artifact metadata is invalid ({type(exc).__name__})") from None
    _validate_metadata(metadata)
    import joblib

    try:
        pipeline = joblib.load(model_path)
    except Exception as exc:
        raise ArtifactError(f"Artifact could not be read ({type(exc).__name__})") from None
    _validate_pipeline(pipeline, metadata)
    classes = tuple(str(label) for label in pipeline.named_steps["clf"].classes_)
    return LoadedArtifact(
        version=str(metadata["dataset_hash"])[:12],
        threshold=float(metadata["confidence_threshold"]),
        classes=classes,
        supported_model_ids=tuple(metadata["supported_model_ids"]),
        pipeline=pipeline,
        metadata=metadata,
    )


def _trusted_file(path: Path, trusted_root: Path) -> Path:
    root = trusted_root.expanduser().resolve()
    candidate = path.expanduser().resolve()
    if not candidate.is_relative_to(root):
        raise ArtifactError("Artifact path is outside the trusted directory")
    if candidate.suffix != ".joblib":
        raise ArtifactError("Artifact must be a .joblib file")
    if not candidate.is_file():
        raise ArtifactError("Artifact file is absent")
    return candidate


def _validate_metadata(metadata: object) -> None:
    if not isinstance(metadata, dict):
        raise ArtifactError("Artifact metadata must be an object")
    if metadata.get("format") != ARTIFACT_FORMAT:
        raise ArtifactError("Artifact format is incompatible")
    if metadata.get("provenance") == "mock":
        raise ArtifactError("Mock artifacts are not trusted routing evidence")
    if metadata.get("evidence") == "fixture":
        raise ArtifactError("Fixture artifacts are for tests and are not trusted for routing")
    versions = metadata.get("library_versions")
    if not isinstance(versions, dict) or "scikit-learn" not in versions:
        raise ArtifactError("Artifact is missing scikit-learn version metadata")
    import sklearn

    saved = str(versions["scikit-learn"]).split(".")
    current = sklearn.__version__.split(".")
    if saved[:2] != current[:2]:
        raise ArtifactError(
            "Artifact scikit-learn "
            f"{versions['scikit-learn']} is incompatible with {sklearn.__version__}"
        )
    threshold = metadata.get("confidence_threshold")
    if not isinstance(threshold, (int, float)) or not 0 <= float(threshold) <= 1:
        raise ArtifactError("Artifact confidence threshold is invalid")
    if metadata.get("features") != "tfidf-prompt-text":
        raise ArtifactError("Artifact features are incompatible")
    if metadata.get("estimator") != "TfidfVectorizer+LogisticRegression":
        raise ArtifactError("Artifact estimator is incompatible")


def _validate_pipeline(pipeline: object, metadata: dict[str, object]) -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    if not isinstance(pipeline, Pipeline):
        raise ArtifactError("Artifact is not a scikit-learn Pipeline")
    steps = dict(pipeline.steps)
    if not isinstance(steps.get("tfidf"), TfidfVectorizer):
        raise ArtifactError("Artifact is missing the TF-IDF step")
    if not isinstance(steps.get("clf"), LogisticRegression):
        raise ArtifactError("Artifact is missing the logistic regression step")
    classes = [str(label) for label in steps["clf"].classes_]
    if classes != list(metadata.get("classes", [])):
        raise ArtifactError("Artifact classes do not match its metadata")
