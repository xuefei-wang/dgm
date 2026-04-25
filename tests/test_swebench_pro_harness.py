import json
import subprocess
from pathlib import Path

from swe_bench import pro_harness as module


def test_official_eval_timeout_returns_metadata(monkeypatch, tmp_path: Path) -> None:
    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(
            cmd=["python", "scripts/run_swebench_pro_eval.py"],
            timeout=123,
            output="stdout tail",
            stderr="stderr tail",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    metadata = module._run_official_eval(
        patch_bundle=tmp_path / "patches.json",
        official_eval_dir=tmp_path / "official_eval",
        dataset_path=tmp_path / "dataset.jsonl",
        eval_source=tmp_path / "evaluator",
        scripts_dir=tmp_path / "scripts",
        dockerhub_username="jefzda",
        max_workers=3,
        use_local_docker=True,
        docker_platform=None,
        block_network=False,
        timeout_sec=123,
    )

    assert metadata["status"] == "timeout"
    assert metadata["timeout_sec"] == 123
    assert metadata["returncode"] is None
    assert metadata["stdout_tail"] == "stdout tail"
    assert metadata["stderr_tail"] == "stderr tail"


def test_partial_eval_results_are_reported_explicitly(tmp_path: Path) -> None:
    out_dname = tmp_path / "predictions"
    official_eval_dir = tmp_path / "official_eval"
    out_dname.mkdir()
    official_eval_dir.mkdir()

    success_json = out_dname / "instance-success.json"
    unresolved_json = out_dname / "instance-unresolved.json"
    timeout_json = out_dname / "instance-timeout.json"
    for path, instance_id in [
        (success_json, "instance-success"),
        (unresolved_json, "instance-unresolved"),
        (timeout_json, "instance-timeout"),
    ]:
        path.write_text(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "model_patch": "diff --git a/a b/a\n",
                    "eval_result": "pending_eval",
                    "success": True,
                }
            ),
            encoding="utf-8",
        )

    success_output_dir = official_eval_dir / "instance-success"
    success_output_dir.mkdir()
    (success_output_dir / "prefix_output.json").write_text(
        json.dumps({"tests": [{"name": "test_a", "status": "PASSED"}]}),
        encoding="utf-8",
    )
    unresolved_output_dir = official_eval_dir / "instance-unresolved"
    unresolved_output_dir.mkdir()
    (unresolved_output_dir / "prefix_output.json").write_text(
        json.dumps({"tests": [{"name": "test_c", "status": "FAILED"}]}),
        encoding="utf-8",
    )

    results = [
        {
            "success": True,
            "instance_id": "instance-success",
            "json_path": str(success_json),
            "model_patch": "diff --git a/a b/a\n",
            "eval_patch": "diff --git a/a b/a\n",
        },
        {
            "success": True,
            "instance_id": "instance-unresolved",
            "json_path": str(unresolved_json),
            "model_patch": "diff --git a/a b/a\n",
            "eval_patch": "diff --git a/a b/a\n",
        },
        {
            "success": True,
            "instance_id": "instance-timeout",
            "json_path": str(timeout_json),
            "model_patch": "diff --git a/a b/a\n",
            "eval_patch": "diff --git a/a b/a\n",
        },
    ]
    entries_by_id = {
        "instance-success": {"fail_to_pass": '["test_a"]', "pass_to_pass": "[]"},
        "instance-unresolved": {"fail_to_pass": '["test_c"]', "pass_to_pass": "[]"},
        "instance-timeout": {"fail_to_pass": '["test_b"]', "pass_to_pass": "[]"},
    }
    partial_eval_results = module._load_partial_eval_results(
        official_eval_dir,
        prefix="prefix",
        entries_by_id=entries_by_id,
    )
    eval_results = dict(partial_eval_results)
    eval_metadata = {
        "status": "timeout",
        "timeout_sec": 600,
        "returncode": None,
        "stdout_tail": "evaluator stdout",
        "stderr_tail": "evaluator stderr",
    }

    module._write_eval_logs(
        entries_by_id=entries_by_id,
        results=results,
        out_dname=out_dname,
        official_eval_dir=official_eval_dir,
        prefix="prefix",
        eval_results=eval_results,
        eval_metadata=eval_metadata,
    )
    report = module.build_report(
        list(entries_by_id.values()),
        results,
        eval_results,
        out_dname=out_dname,
        eval_metadata=eval_metadata,
    )

    success_prediction = json.loads(success_json.read_text(encoding="utf-8"))
    unresolved_prediction = json.loads(unresolved_json.read_text(encoding="utf-8"))
    timeout_prediction = json.loads(timeout_json.read_text(encoding="utf-8"))
    timeout_eval_log = (out_dname / "instance-timeout_eval.md").read_text(encoding="utf-8")

    assert success_prediction["eval_result"] == "resolved"
    assert unresolved_prediction["eval_result"] == "unresolved"
    assert timeout_prediction["eval_result"] == "eval_timeout"
    assert "timed out after 600 seconds" in timeout_eval_log
    assert "evaluator stderr" in timeout_eval_log
    assert report["official_eval_status"] == "timeout"
    assert report["eval_incomplete_instances"] == 1
    assert report["eval_incomplete_ids"] == ["instance-timeout"]


def test_agent_timeout_without_patch_is_reported_explicitly(tmp_path: Path) -> None:
    out_dname = tmp_path / "predictions"
    official_eval_dir = tmp_path / "official_eval"
    out_dname.mkdir()
    official_eval_dir.mkdir()

    timeout_json = out_dname / "instance-timeout.json"
    timeout_json.write_text(
        json.dumps(
            {
                "instance_id": "instance-timeout",
                "model_patch": "",
                "eval_result": "agent_timeout",
                "success": True,
                "agent_status": "timeout",
                "agent_exit_code": 124,
            }
        ),
        encoding="utf-8",
    )

    results = [
        {
            "success": True,
            "instance_id": "instance-timeout",
            "json_path": str(timeout_json),
            "model_patch": "",
            "agent_status": "timeout",
            "agent_exit_code": 124,
        }
    ]
    entries_by_id = {"instance-timeout": {"fail_to_pass": "[]", "pass_to_pass": "[]"}}

    module._write_eval_logs(
        entries_by_id=entries_by_id,
        results=results,
        out_dname=out_dname,
        official_eval_dir=official_eval_dir,
        prefix="prefix",
        eval_results={},
        eval_metadata={"status": "skipped", "timeout_sec": 600},
    )
    report = module.build_report(
        [entries_by_id["instance-timeout"]],
        results,
        {},
        out_dname=out_dname,
        eval_metadata={"status": "skipped", "timeout_sec": 600},
    )

    prediction = json.loads(timeout_json.read_text(encoding="utf-8"))
    timeout_eval_log = (out_dname / "instance-timeout_eval.md").read_text(encoding="utf-8")

    assert prediction["eval_result"] == "agent_timeout"
    assert "prediction generation timed out" in timeout_eval_log
    assert report["agent_timeout_instances"] == 1
    assert report["agent_timeout_ids"] == ["instance-timeout"]
    assert report["empty_patch_instances"] == 0
