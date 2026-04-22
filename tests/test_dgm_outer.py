import json
from types import SimpleNamespace

import pytest

from DGM_outer import choose_selfimproves, validate_swebench_pro_paths
from self_improve_step import _ensure_container_git_repo, _read_container_head_commit


def test_choose_selfimproves_handles_all_empty_patch_parent(tmp_path):
    parent_dir = tmp_path / "initial"
    parent_dir.mkdir()
    (parent_dir / "metadata.json").write_text(
        json.dumps(
            {
                "overall_performance": {
                    "accuracy_score": 0.0,
                    "total_unresolved_ids": [],
                    "total_emptypatch_ids": ["instance-empty"],
                    "total_resolved_ids": [],
                }
            }
        ),
        encoding="utf-8",
    )

    entries = choose_selfimproves(str(tmp_path), ["initial"], 1)

    assert entries == [("initial", "solve_empty_patches")]


def test_validate_swebench_pro_paths_fails_fast_for_missing_eval_source(tmp_path):
    dataset = tmp_path / "test.jsonl"
    task_map = tmp_path / "task_map.json"
    dataset.write_text("{}", encoding="utf-8")
    task_map.write_text("{}", encoding="utf-8")

    args = SimpleNamespace(
        swebench_pro_dataset_path=str(dataset),
        swebench_pro_task_map=str(task_map),
        swebench_pro_eval_source=str(tmp_path / "missing-evaluator"),
        swebench_pro_scripts_dir=None,
    )

    with pytest.raises(FileNotFoundError, match="evaluator script"):
        validate_swebench_pro_paths(args)


class _FakeExecResult:
    exit_code = 0
    output = b""


class _FakeContainer:
    def __init__(self):
        self.calls = []

    def exec_run(self, cmd, workdir=None):
        self.calls.append((cmd, workdir))
        return _FakeExecResult()


def test_ensure_container_git_repo_recovers_copied_submodule_gitdir():
    container = _FakeContainer()

    _ensure_container_git_repo(container)

    command = (
        "/bin/sh -c '"
        "if ! git -C /dgm rev-parse --is-inside-work-tree >/dev/null 2>&1; then "
        "rm -rf /dgm/.git && git -C /dgm init; "
        "fi'"
    )
    assert container.calls == [
        (command, "/")
    ]


def test_read_container_head_commit_uses_rev_parse_output():
    container = _FakeContainer()

    def exec_run(cmd, workdir=None):
        container.calls.append((cmd, workdir))
        result = _FakeExecResult()
        result.output = b"abc123def456\n"
        return result

    container.exec_run = exec_run

    assert _read_container_head_commit(container) == "abc123def456"
    assert container.calls == [("git rev-parse HEAD", "/dgm/")]
