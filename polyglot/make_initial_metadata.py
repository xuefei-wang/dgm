#!/usr/bin/env python3
"""Create DGM initial metadata from a Polyglot harness report."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _duplicate_ids(values):
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _validated_report_ids(report: dict, field: str, expected: set[str]) -> list[str]:
    values = report.get(field, [])
    duplicate_ids = _duplicate_ids(values)
    if duplicate_ids:
        raise ValueError(f"Report {field} contains duplicate IDs: {duplicate_ids[:10]}")

    extra_ids = sorted(set(values) - expected)
    if extra_ids:
        raise ValueError(f"Report {field} contains IDs outside task map: {extra_ids[:10]}")

    return sorted(values)


def _prediction_task_id(path: Path) -> str | None:
    name = path.name
    if name.endswith("_eval.md"):
        return name[: -len("_eval.md")]
    if name.endswith(".json"):
        return name[:-len(".json")]
    if name.endswith(".md"):
        return name[:-len(".md")]
    return None


def validate_predictions_dir(predictions_dir: Path, task_ids: list[str]) -> Path:
    if not predictions_dir.is_dir():
        raise FileNotFoundError(f"Predictions directory does not exist: {predictions_dir}")

    run_dirs = sorted(path for path in predictions_dir.iterdir() if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No prediction run directories found under {predictions_dir}")
    if len(run_dirs) != 1:
        raise ValueError(
            "Expected exactly one prediction run directory for Polyglot initial metadata, "
            f"found {len(run_dirs)} under {predictions_dir}"
        )

    expected = set(task_ids)
    run_dir = run_dirs[0]
    missing = []
    for task_id in task_ids:
        required_files = (
            run_dir / f"{task_id}.json",
            run_dir / f"{task_id}.md",
            run_dir / f"{task_id}_eval.md",
        )
        if not all(path.is_file() for path in required_files):
            missing.append(task_id)

    if missing:
        raise FileNotFoundError(
            "Missing prediction JSON, agent log, or eval log for Polyglot tasks: "
            f"{missing[:10]}"
        )

    extra_files = []
    for path in sorted(run_dir.iterdir()):
        if not path.is_file():
            continue
        task_id = _prediction_task_id(path)
        if task_id is not None and task_id not in expected:
            extra_files.append(path.name)

    if extra_files:
        raise ValueError(
            "Predictions directory contains files outside task map: "
            f"{extra_files[:10]}"
        )

    return run_dir


def package_predictions_dir(predictions_dir: Path, task_ids: list[str], output_dir: Path) -> None:
    run_dir = validate_predictions_dir(predictions_dir, task_ids)
    packaged_run_dir = output_dir / "predictions" / run_dir.name
    packaged_run_dir.mkdir(parents=True, exist_ok=True)

    for task_id in task_ids:
        for suffix in (".json", ".md", "_eval.md"):
            source = run_dir / f"{task_id}{suffix}"
            shutil.copy2(source, packaged_run_dir / source.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="Polyglot harness report JSON.")
    parser.add_argument("--task-map", type=Path, required=True, help="Task map used for the harness run.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where metadata.json will be written.")
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        help="Optional harness predictions directory to package for DGM diagnosis.",
    )
    args = parser.parse_args()

    report = load_json(args.report.resolve())
    task_ids = load_json(args.task_map.resolve())
    duplicate_task_ids = _duplicate_ids(task_ids)
    if duplicate_task_ids:
        raise ValueError(f"Task map contains duplicate IDs: {duplicate_task_ids[:10]}")

    expected = set(task_ids)
    submitted_ids = _validated_report_ids(report, "submitted_ids", expected)
    submitted = set(submitted_ids)
    if submitted != expected:
        missing = sorted(expected - submitted)
        extra = sorted(submitted - expected)
        raise ValueError(f"Report submitted IDs do not match task map. missing={missing[:10]} extra={extra[:10]}")

    resolved_ids = _validated_report_ids(report, "resolved_ids", expected)
    unresolved_ids = _validated_report_ids(report, "unresolved_ids", expected)
    empty_patch_ids = _validated_report_ids(report, "empty_patch_ids", expected)
    incomplete_ids = _validated_report_ids(report, "incomplete_ids", expected)
    error_ids = _validated_report_ids(report, "error_ids", expected)

    # DGM samples self-improvement entries from unresolved and empty-patch IDs.
    # Treat incomplete/error runs as unresolved so the initial parent still covers
    # exactly the benchmark task map denominator.
    total_unresolved_ids = sorted(set(unresolved_ids + incomplete_ids + error_ids))
    submitted_instances = len(task_ids)
    resolved_instances = len(resolved_ids)
    accuracy_score = resolved_instances / submitted_instances if submitted_instances else 0.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    packaged_report = args.output_dir / args.report.name
    shutil.copy2(args.report.resolve(), packaged_report)
    if args.predictions_dir:
        package_predictions_dir(args.predictions_dir.resolve(), task_ids, args.output_dir)

    metadata = {
        "run_id": "initial",
        "overall_performance": {
            "accuracy_score": accuracy_score,
            "total_resolved_instances": resolved_instances,
            "total_submitted_instances": submitted_instances,
            "files": [str(packaged_report.resolve())],
            "total_unresolved_ids": total_unresolved_ids,
            "total_emptypatch_ids": empty_patch_ids,
            "total_resolved_ids": resolved_ids,
        },
    }

    output_path = args.output_dir / "metadata.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=4)
        handle.write("\n")
    print(f"Wrote {output_path}")
    print(
        f"Initial score: {resolved_instances}/{submitted_instances} "
        f"({accuracy_score:.3f}); unresolved={len(total_unresolved_ids)} "
        f"empty={len(empty_patch_ids)}"
    )


if __name__ == "__main__":
    main()
