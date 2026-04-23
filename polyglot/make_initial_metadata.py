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

    submitted_ids = report.get("submitted_ids", [])
    duplicate_submitted_ids = _duplicate_ids(submitted_ids)
    if duplicate_submitted_ids:
        raise ValueError(f"Report submitted_ids contains duplicate IDs: {duplicate_submitted_ids[:10]}")

    duplicate_resolved_ids = _duplicate_ids(report.get("resolved_ids", []))
    if duplicate_resolved_ids:
        raise ValueError(f"Report resolved_ids contains duplicate IDs: {duplicate_resolved_ids[:10]}")

    expected = set(task_ids)
    submitted = set(submitted_ids)
    if submitted != expected:
        missing = sorted(expected - submitted)
        extra = sorted(submitted - expected)
        raise ValueError(f"Report submitted IDs do not match task map. missing={missing[:10]} extra={extra[:10]}")

    resolved_ids = sorted(report.get("resolved_ids", []))
    unresolved_ids = sorted(report.get("unresolved_ids", []))
    empty_patch_ids = sorted(report.get("empty_patch_ids", []))
    incomplete_ids = sorted(report.get("incomplete_ids", []))
    error_ids = sorted(report.get("error_ids", []))

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
        packaged_predictions = args.output_dir / "predictions"
        shutil.copytree(args.predictions_dir.resolve(), packaged_predictions, dirs_exist_ok=True)

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
