import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from prompts import self_improvement_prompt
from swe_bench.pro_harness import (
    _build_test_description,
    _copy_dgm_runtime,
    _raise_for_agent_failure,
    build_report,
    load_task_ids,
)


def test_swebench_pro_diagnosis_prompt_does_not_include_private_artifacts(monkeypatch):
    monkeypatch.setattr(
        self_improvement_prompt,
        "find_selfimprove_eval_logs",
        lambda *args, **kwargs: (["agent log"], ["eval log"], ["predicted patch"], ["unresolved"]),
    )
    monkeypatch.setattr(self_improvement_prompt, "get_current_code", lambda *args, **kwargs: "agent code")
    dataset = [
        {
            "instance_id": "instance_public",
            "repo": "owner/repo",
            "repo_language": "python",
            "problem_statement": "public problem",
            "requirements": "public requirements",
            "interface": "public interface",
            "patch": "SECRET_GOLD_PATCH",
            "test_patch": "SECRET_PRIVATE_TEST_PATCH",
        }
    ]

    _, prompt = self_improvement_prompt.get_diagnose_prompt_swebench_pro(
        "instance_public",
        "initial",
        ".",
        ".",
        dataset,
    )

    assert "public problem" in prompt
    assert "predicted patch" in prompt
    assert "SECRET_GOLD_PATCH" not in prompt
    assert "SECRET_PRIVATE_TEST_PATCH" not in prompt


def test_prediction_prompt_does_not_expose_official_eval_scripts():
    description = _build_test_description(
        {
            "selected_test_files_to_run": ["secret/private_test.py"],
            "before_repo_set_cmd": "bash /workspace/run_script.sh",
        }
    )

    assert "/workspace/run_script.sh" not in description
    assert "secret/private_test.py" not in description
    assert "public files and tests" in description


def test_copy_dgm_runtime_does_not_copy_official_eval_artifacts(monkeypatch):
    copied_destinations = []

    def fake_copy_to_container(_container, _source, dest):
        copied_destinations.append(dest)

    monkeypatch.setitem(
        sys.modules,
        "swe_bench.utils",
        SimpleNamespace(copy_to_container=fake_copy_to_container),
    )

    class FakeContainer:
        def __init__(self):
            self.commands = []

        def exec_run(self, command, workdir="/"):
            self.commands.append((command, workdir))
            return SimpleNamespace(exit_code=0, output=b"")

    container = FakeContainer()
    _copy_dgm_runtime(container)

    assert "/workspace/run_script.sh" not in copied_destinations
    assert "/workspace/parser.py" not in copied_destinations
    assert container.commands == [("mkdir -p /dgm", "/")]


def test_agent_nonzero_exit_becomes_error_prediction():
    with pytest.raises(RuntimeError, match="exit code 2"):
        _raise_for_agent_failure(SimpleNamespace(exit_code=2, output=b"agent crashed"))


def test_build_report_marks_missing_eval_results_as_errors():
    entries = [{"instance_id": "instance_a"}, {"instance_id": "instance_b"}]
    results = [
        {"success": True, "instance_id": "instance_a", "model_patch": "diff --git a/a b/a\n", "json_path": "a.json"},
        {"success": True, "instance_id": "instance_b", "model_patch": "diff --git a/b b/b\n", "json_path": "b.json"},
    ]

    report = build_report(entries, results, {"instance_a": True})

    assert report["resolved_ids"] == ["instance_a"]
    assert report["unresolved_ids"] == ["instance_b"]
    assert report["error_ids"] == ["instance_b"]
    assert report["error_instances"] == 1


def test_load_task_ids_accepts_task_objects(tmp_path):
    task_map = tmp_path / "tasks.json"
    task_map.write_text('{"tasks": [{"task_id": "instance_a"}, {"instance_id": "instance_b"}]}', encoding="utf-8")

    assert load_task_ids(Path(task_map)) == ["instance_a", "instance_b"]
