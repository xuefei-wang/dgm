from pathlib import Path

from prompts import self_improvement_prompt
from swe_bench.pro_harness import build_report, load_task_ids


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


def test_build_report_marks_missing_eval_results_as_errors():
    entries = [{"instance_id": "instance_a"}, {"instance_id": "instance_b"}]
    results = [
        {"success": True, "instance_id": "instance_a", "model_patch": "diff --git a/a b/a\n", "json_path": "a.json"},
        {"success": True, "instance_id": "instance_b", "model_patch": "diff --git a/b b/b\n", "json_path": "b.json"},
    ]

    report = build_report(entries, results, {"instance_a": True})

    assert report["resolved_ids"] == ["instance_a"]
    assert report["unresolved_ids"] == []
    assert report["error_ids"] == ["instance_b"]
    assert report["error_instances"] == 1


def test_load_task_ids_accepts_task_objects(tmp_path):
    task_map = tmp_path / "tasks.json"
    task_map.write_text('{"tasks": [{"task_id": "instance_a"}, {"instance_id": "instance_b"}]}', encoding="utf-8")

    assert load_task_ids(Path(task_map)) == ["instance_a", "instance_b"]
