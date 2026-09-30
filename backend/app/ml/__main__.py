"""Commands for the optional ML router. None of these call a provider."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.ml.artifact import load_trusted_artifact, save_artifact
from app.ml.compare import compare_policies
from app.ml.labels import label_dataset
from app.ml.schema import TrainingDataError, dataset_hash, load_training_dataset
from app.ml.split import SplitError, assign_splits
from app.ml.train import fit_router


def main(argv: list[str] | None = None) -> int:
    """CLI entry. Training refuses mock data and unlabeled smoke files."""
    received = list(sys.argv[1:] if argv is None else argv)
    if received and received[0] == "collect":
        from app.ml.collect import run_collect

        return run_collect(received[1:])
    parser = argparse.ArgumentParser(
        description=(
            "Train and compare the optional ML router. "
            "Live collection is `python -m app.ml collect --help`."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Check a training dataset file.")
    validate.add_argument("--dataset", required=True)
    label = sub.add_parser("label", help="Write supervised labels and exclusion counts.")
    label.add_argument("--dataset", required=True)
    label.add_argument("--output", required=True)
    train = sub.add_parser("train", help="Fit the TF-IDF logistic regression router.")
    train.add_argument("--dataset", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--seed", type=int, default=17)
    compare = sub.add_parser("compare", help="Compare policies on the held-out split.")
    compare.add_argument("--dataset", required=True)
    compare.add_argument("--artifact", required=True)
    compare.add_argument("--trusted-root", default=None)
    compare.add_argument("--seed", type=int, default=17)
    compare.add_argument("--quality-tolerance", type=float, default=None)
    compare.add_argument("--cost-objective", choices=["min_mean_cost"], default=None)
    compare.add_argument("--output", default=None)
    args = parser.parse_args(received)
    try:
        if args.command == "validate":
            return _validate(Path(args.dataset))
        if args.command == "label":
            return _label(Path(args.dataset), Path(args.output))
        if args.command == "train":
            return _train(Path(args.dataset), Path(args.output_dir), args.seed)
        return _compare(args)
    except (TrainingDataError, SplitError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _validate(path: Path) -> int:
    dataset = load_training_dataset(path)
    if dataset.provenance == "mock":
        print("Mock provenance is not training evidence.", file=sys.stderr)
        return 2
    report = label_dataset(dataset)
    print(
        f"Dataset {dataset.version} provenance={dataset.provenance} "
        f"evidence={dataset.evidence} prompts={len(dataset.prompts)} "
        f"labeled={report.included} excluded={report.excluded}"
    )
    if dataset.provenance == "mock" or dataset.evidence == "fixture":
        print("This file is not measured training evidence.")
    if report.exploratory:
        print("Labeled count is exploratory. Do not treat it as a promotion result.")
    return 0


def _label(path: Path, output: Path) -> int:
    dataset = load_training_dataset(path)
    report = label_dataset(dataset)
    payload = report.to_dict()
    payload["dataset_hash"] = dataset_hash(dataset)
    payload["evidence"] = dataset.evidence
    payload["provenance"] = dataset.provenance
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"Labeled {report.included}; excluded {report.excluded}; "
        f"wrote {output}"
    )
    return 0


def _train(path: Path, output_dir: Path, seed: int) -> int:
    dataset = load_training_dataset(path)
    if dataset.evidence != "measured" or dataset.provenance != "live":
        print(
            "Refusing to train. Mock and fixture datasets are not measured "
            "training evidence. Collect live per-model evaluations first.",
            file=sys.stderr,
        )
        return 2
    report = label_dataset(dataset)
    digest = dataset_hash(dataset)
    assignment = assign_splits(report.prompts, seed=seed, dataset_hash=digest)
    trained = fit_router(assignment, seed=seed)
    model_path = save_artifact(
        output_dir,
        trained=trained,
        dataset_hash=digest,
        split=assignment.to_dict(),
        seed=seed,
        supported_model_ids=tuple(item.model_id for item in dataset.candidates),
        evidence=dataset.evidence,
        provenance=dataset.provenance,
    )
    print(f"Wrote {model_path}")
    if assignment.exploratory:
        print(assignment.exploratory_reason)
        print("The artifact is exploratory and is not evidence for enabling ML routing.")
    return 0


def _compare(args: argparse.Namespace) -> int:
    dataset = load_training_dataset(Path(args.dataset))
    report = label_dataset(dataset)
    digest = dataset_hash(dataset)
    assignment = assign_splits(report.prompts, seed=args.seed, dataset_hash=digest)
    root = Path(args.trusted_root) if args.trusted_root else Path(args.artifact).resolve().parent
    artifact = load_trusted_artifact(Path(args.artifact), trusted_root=root)
    comparison = compare_policies(
        dataset,
        report.prompts,
        assignment,
        artifact,
        quality_tolerance=args.quality_tolerance,
        cost_objective=args.cost_objective,
    )
    text = json.dumps(comparison, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text)
    winner = comparison["promotion"]["winner"]
    print(f"Promotion winner: {winner if winner else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
