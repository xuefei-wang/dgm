#!/usr/bin/env python3
"""Run the DGM Polyglot initial agent on an explicit task map."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.common_utils import load_json_file

from polyglot.harness import harness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True, help="Polyglot benchmark metadata JSON.")
    parser.add_argument("--task-map", type=Path, required=True, help="Explicit Polyglot task ID list.")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Prediction artifact directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Report output directory.")
    parser.add_argument("--model-name", default="initial_polyglot_50", help="Report/model prefix.")
    parser.add_argument("--max-workers", type=int, default=5, help="Parallel task workers.")
    args = parser.parse_args()

    dataset_path = args.dataset_path.resolve()
    task_map = load_json_file(str(args.task_map.resolve()))
    if not isinstance(task_map, list):
        raise ValueError(f"Expected task map to be a JSON list: {args.task_map}")

    os.environ["DGM_POLYGLOT_METADATA"] = str(dataset_path)
    harness(
        dataset_path=str(dataset_path),
        test_task_list=task_map,
        num_samples=-1,
        max_workers=args.max_workers,
        model_name_or_path=args.model_name,
        model_patch_paths=None,
        num_evals=1,
        num_evals_parallel=1,
        pred_dname=str(args.pred_dir.resolve()),
        output_dir=str(args.output_dir.resolve()),
    )


if __name__ == "__main__":
    main()
