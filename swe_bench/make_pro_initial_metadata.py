#!/usr/bin/env python3
"""Create DGM initial metadata from a SWE-bench Pro harness report."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

DGM_ROOT = Path(__file__).resolve().parents[1]
if str(DGM_ROOT) not in sys.path:
    sys.path.insert(0, str(DGM_ROOT))

from swe_bench.pro_harness import DEFAULT_TASK_MAP, load_task_ids, safe_instance_filename


def _load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_metadata(report: dict, task_ids: list[str], packaged_report: Path) -> dict:
    expected = set(task_ids)
    submitted = set(report.get("submitted_ids", []))
    if submitted != expected:
        missing = sorted(expected - submitted)
        extra = sorted(submitted - expected)
        raise ValueError(
            "Report submitted IDs do not match task map. "
            f"missing={missing[:10]} extra={extra[:10]}"
        )

    resolved_ids = sorted(report.get("resolved_ids", []))
    unresolved_ids = sorted(report.get("unresolved_ids", []))
    empty_patch_ids = sorted(report.get("empty_patch_ids", []))
    incomplete_ids = sorted(report.get("incomplete_ids", []))
    error_ids = sorted(report.get("error_ids", []))

    total_unresolved_ids = sorted(set(unresolved_ids + incomplete_ids + error_ids))
    submitted_instances = int(report.get("submitted_instances", len(task_ids)))
    resolved_instances = len(resolved_ids)
    accuracy_score = resolved_instances / submitted_instances if submitted_instances else 0.0

    return {
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


def validate_predictions_dir(predictions_dir: Path, task_ids: list[str]) -> None:
    if not predictions_dir.is_dir():
        raise FileNotFoundError(f"Predictions directory does not exist: {predictions_dir}")

    run_dirs = sorted(path for path in predictions_dir.iterdir() if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No prediction run directories found under {predictions_dir}")

    missing = []
    for task_id in task_ids:
        filename = safe_instance_filename(task_id)
        found_complete_run = any(
            (run_dir / f"{filename}.json").is_file()
            and (run_dir / f"{filename}.md").is_file()
            and (run_dir / f"{filename}_eval.md").is_file()
            for run_dir in run_dirs
        )
        if not found_complete_run:
            missing.append(task_id)

    if missing:
        raise FileNotFoundError(
            "Missing prediction JSON, agent log, or eval log for SWE-bench Pro tasks: "
            f"{missing[:10]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="SWE-bench Pro harness report JSON.")
    parser.add_argument("--task-map", type=Path, default=DEFAULT_TASK_MAP, help="Task map used for the harness run.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where metadata.json will be written.")
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="Harness predictions directory to package for DGM diagnosis.",
    )
    args = parser.parse_args()

    report = _load_json(args.report.resolve())
    task_ids = load_task_ids(args.task_map.resolve())
    if task_ids is None:
        raise ValueError(f"No task IDs loaded from {args.task_map}")
    validate_predictions_dir(args.predictions_dir.resolve(), task_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    packaged_report = args.output_dir / args.report.name
    shutil.copy2(args.report.resolve(), packaged_report)
    packaged_predictions = args.output_dir / "predictions"
    shutil.copytree(args.predictions_dir.resolve(), packaged_predictions, dirs_exist_ok=True)

    metadata = build_metadata(report, task_ids, packaged_report)
    output_path = args.output_dir / "metadata.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=4)
        handle.write("\n")

    perf = metadata["overall_performance"]
    print(f"Wrote {output_path}")
    print(
        f"Initial score: {perf['total_resolved_instances']}/{perf['total_submitted_instances']} "
        f"({perf['accuracy_score']:.3f}); "
        f"unresolved={len(perf['total_unresolved_ids'])} "
        f"empty={len(perf['total_emptypatch_ids'])}"
    )


if __name__ == "__main__":
    main()
