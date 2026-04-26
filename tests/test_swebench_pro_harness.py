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


def test_partial_eval_recovers_resolved_when_batch_times_out_but_per_instance_output_survives(
    tmp_path: Path,
) -> None:
    """When the batch evaluator times out, instances whose per-instance
    `<prefix>_output.json` file survived must be recovered as
    resolved/unresolved (not eval_timeout). This is the entire point of the
    partial-eval feature."""
    out_dname = tmp_path / "predictions"
    official_eval_dir = tmp_path / "official_eval"
    out_dname.mkdir()
    official_eval_dir.mkdir()

    recovered_json = out_dname / "instance-recovered.json"
    recovered_json.write_text(
        json.dumps(
            {
                "instance_id": "instance-recovered",
                "model_patch": "diff --git a/a b/a\n",
                "eval_result": "pending_eval",
                "success": True,
            }
        ),
        encoding="utf-8",
    )

    recovered_dir = official_eval_dir / "instance-recovered"
    recovered_dir.mkdir()
    (recovered_dir / "prefix_output.json").write_text(
        json.dumps({"tests": [{"name": "test_a", "status": "PASSED"}]}),
        encoding="utf-8",
    )

    results = [
        {
            "success": True,
            "instance_id": "instance-recovered",
            "json_path": str(recovered_json),
            "model_patch": "diff --git a/a b/a\n",
            "eval_patch": "diff --git a/a b/a\n",
        },
    ]
    entries_by_id = {"instance-recovered": {"fail_to_pass": '["test_a"]', "pass_to_pass": "[]"}}

    partial = module._load_partial_eval_results(
        official_eval_dir,
        prefix="prefix",
        entries_by_id=entries_by_id,
    )
    eval_results = dict(partial)
    eval_metadata = {
        "status": "timeout",
        "timeout_sec": 600,
        "returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
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

    prediction = json.loads(recovered_json.read_text(encoding="utf-8"))
    assert prediction["eval_result"] == "resolved"
    assert report["resolved_instances"] == 1
    # Even though the BATCH eval timed out, this recovered instance must
    # NOT be counted as eval_incomplete.
    assert report["eval_incomplete_instances"] == 0
    assert report["unresolved_instances"] == 0


def test_eval_error_status_marks_instance_eval_error(tmp_path: Path) -> None:
    """When the official evaluator subprocess exits non-zero (eval_error
    branch — distinct from timeout), instances without a per-instance
    output must be classified eval_error and the diagnostic must include
    the returncode."""
    out_dname = tmp_path / "predictions"
    official_eval_dir = tmp_path / "official_eval"
    out_dname.mkdir()
    official_eval_dir.mkdir()

    err_json = out_dname / "instance-evalerror.json"
    err_json.write_text(
        json.dumps(
            {
                "instance_id": "instance-evalerror",
                "model_patch": "diff --git a/a b/a\n",
                "eval_result": "pending_eval",
                "success": True,
            }
        ),
        encoding="utf-8",
    )

    results = [
        {
            "success": True,
            "instance_id": "instance-evalerror",
            "json_path": str(err_json),
            "model_patch": "diff --git a/a b/a\n",
            "eval_patch": "diff --git a/a b/a\n",
        },
    ]
    entries_by_id = {"instance-evalerror": {"fail_to_pass": '["test_a"]', "pass_to_pass": "[]"}}
    eval_metadata = {
        "status": "failed",
        "timeout_sec": 600,
        "returncode": 137,
        "stdout_tail": "",
        "stderr_tail": "OOM killed",
    }

    module._write_eval_logs(
        entries_by_id=entries_by_id,
        results=results,
        out_dname=out_dname,
        official_eval_dir=official_eval_dir,
        prefix="prefix",
        eval_results={},
        eval_metadata=eval_metadata,
    )
    report = module.build_report(
        list(entries_by_id.values()),
        results,
        {},
        out_dname=out_dname,
        eval_metadata=eval_metadata,
    )

    prediction = json.loads(err_json.read_text(encoding="utf-8"))
    eval_log = (out_dname / "instance-evalerror_eval.md").read_text(encoding="utf-8")
    assert prediction["eval_result"] == "eval_error"
    assert "return code 137" in eval_log
    assert "OOM killed" in eval_log
    assert report["eval_incomplete_instances"] == 1
    # eval_error / eval_timeout / eval_incomplete must NOT inflate
    # unresolved_instances (regression guard for the unresolved_ids fix).
    assert report["unresolved_instances"] == 0


def test_unresolved_excludes_incomplete_eval_instances(tmp_path: Path) -> None:
    """Regression guard: unresolved_instances must not double-count
    instances whose eval timed out / errored / never produced a result.
    Previously, `not eval_results.get(id, False)` was True for missing
    keys, inflating unresolved alongside eval_incomplete."""
    out_dname = tmp_path / "predictions"
    official_eval_dir = tmp_path / "official_eval"
    out_dname.mkdir()
    official_eval_dir.mkdir()

    paths = []
    results = []
    entries_by_id = {}
    for instance_id in ["a", "b", "c"]:
        path = out_dname / f"{instance_id}.json"
        path.write_text(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "model_patch": "diff --git a/x b/x\n",
                    "eval_result": "pending_eval",
                    "success": True,
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
        results.append(
            {
                "success": True,
                "instance_id": instance_id,
                "json_path": str(path),
                "model_patch": "diff --git a/x b/x\n",
                "eval_patch": "diff --git a/x b/x\n",
            }
        )
        entries_by_id[instance_id] = {"fail_to_pass": '["t"]', "pass_to_pass": "[]"}

    # Empty eval_results — batch eval timed out before producing anything.
    report = module.build_report(
        list(entries_by_id.values()),
        results,
        {},  # no eval_results entries
        out_dname=out_dname,
        eval_metadata={"status": "timeout", "timeout_sec": 600},
    )
    assert report["resolved_instances"] == 0
    assert report["unresolved_instances"] == 0  # was 3 before the fix
    assert report["eval_incomplete_instances"] == 3


def test_agent_timeout_with_non_empty_patch_does_not_double_count(tmp_path: Path) -> None:
    """Regression guard: if the agent times out but emits a non-empty
    patch, the instance should land in attempted_eval (and resolved or
    unresolved or eval_*) — NOT in agent_timeout_ids, which is reserved
    for timed-out runs that produced no usable patch."""
    out_dname = tmp_path / "predictions"
    out_dname.mkdir()
    official_eval_dir = tmp_path / "official_eval"
    official_eval_dir.mkdir()

    json_path = out_dname / "x.json"
    json_path.write_text(
        json.dumps(
            {
                "instance_id": "x",
                "model_patch": "diff --git a/x b/x\n",
                "eval_result": "pending_eval",
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
            "instance_id": "x",
            "json_path": str(json_path),
            "model_patch": "diff --git a/x b/x\n",
            "eval_patch": "diff --git a/x b/x\n",
            "agent_status": "timeout",
            "agent_exit_code": 124,
        }
    ]
    report = module.build_report(
        [{"fail_to_pass": '["t"]', "pass_to_pass": "[]"}],
        results,
        {"x": True},  # eval succeeded despite agent timeout
        out_dname=out_dname,
        eval_metadata={"status": "ok", "timeout_sec": 600, "returncode": 0},
    )
    assert report["resolved_instances"] == 1
    # Patch was non-empty so this is NOT an agent_timeout outcome.
    assert report["agent_timeout_instances"] == 0
    assert report["empty_patch_instances"] == 0


def test_resume_propagates_agent_status_so_overwrite_does_not_corrupt(
    tmp_path: Path,
) -> None:
    """Regression guard: process_entry's resume early-return must surface
    agent_status from the existing JSON so _write_eval_logs reaches the
    same branch on resume that it did on the original run. Previously
    agent_status was dropped, causing agent_timeout to be overwritten to
    empty_patch on resume."""
    out_dname = tmp_path / "predictions"
    out_dname.mkdir()
    official_eval_dir = tmp_path / "official_eval"
    official_eval_dir.mkdir()

    json_path = out_dname / "x.json"
    json_path.write_text(
        json.dumps(
            {
                "instance_id": "x",
                "model_patch": "",
                "eval_result": "agent_timeout",
                "success": True,
                "agent_status": "timeout",
                "agent_exit_code": 124,
            }
        ),
        encoding="utf-8",
    )
    # Simulate resume: the result dict should carry agent_status, mirroring
    # the post-fix process_entry early-return.
    existing = json.loads(json_path.read_text(encoding="utf-8"))
    resumed = {
        "success": True,
        "instance_id": "x",
        "model_patch": "",
        "json_path": str(json_path),
        "agent_status": existing["agent_status"],
        "agent_exit_code": existing["agent_exit_code"],
    }

    module._write_eval_logs(
        entries_by_id={"x": {"fail_to_pass": "[]", "pass_to_pass": "[]"}},
        results=[resumed],
        out_dname=out_dname,
        official_eval_dir=official_eval_dir,
        prefix="prefix",
        eval_results={},
        eval_metadata={"status": "skipped", "timeout_sec": 600},
    )
    # Must remain agent_timeout, not be overwritten to empty_patch.
    prediction = json.loads(json_path.read_text(encoding="utf-8"))
    assert prediction["eval_result"] == "agent_timeout"


def test_update_prediction_is_atomic_under_simulated_partial_write(
    monkeypatch, tmp_path: Path
) -> None:
    """If write_text fails mid-write (e.g., SIGKILL), the original
    prediction JSON must remain intact — it is also the resume checkpoint.
    The fix uses a tmp file + os.replace, so a failed tmp write cannot
    corrupt the destination."""
    path = tmp_path / "x.json"
    original = {"instance_id": "x", "eval_result": "pending_eval", "success": True}
    path.write_text(json.dumps(original, indent=4), encoding="utf-8")

    # Force the temp-file write to fail; the original must be untouched.
    real_write_text = Path.write_text

    def failing_write_text(self, *args, **kwargs):
        if self.suffix == ".tmp":
            raise OSError("simulated mid-write failure")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", failing_write_text)

    try:
        module._update_prediction(path, eval_result="resolved")
    except OSError:
        pass

    # Original file content must be intact.
    surviving = json.loads(path.read_text(encoding="utf-8"))
    assert surviving == original
    # No stray .tmp artifact left behind that could confuse readers.
    assert not (tmp_path / "x.json.tmp").exists()


def test_load_partial_eval_results_skips_corrupted_output(tmp_path: Path) -> None:
    """A truncated `<prefix>_output.json` (mid-write SIGKILL) must be
    silently skipped — the loader's try/except already covers this; this
    is a regression guard."""
    official_eval_dir = tmp_path / "official_eval"
    official_eval_dir.mkdir()
    instance_dir = official_eval_dir / "broken"
    instance_dir.mkdir()
    (instance_dir / "prefix_output.json").write_text(
        '{"tests": [{"name": "test_a", "status":',  # truncated
        encoding="utf-8",
    )
    entries_by_id = {"broken": {"fail_to_pass": '["test_a"]', "pass_to_pass": "[]"}}
    partial = module._load_partial_eval_results(
        official_eval_dir,
        prefix="prefix",
        entries_by_id=entries_by_id,
    )
    assert partial == {}


def test_env_positive_int_rejects_invalid(monkeypatch) -> None:
    """`_env_positive_int` must reject empty / non-numeric / non-positive
    values with a clear ValueError, instead of crashing inside int() at
    module load with a confusing traceback."""
    monkeypatch.setenv("DGM_TEST_VAR", "")
    # Empty string falls back to default, not an error.
    assert module._env_positive_int("DGM_TEST_VAR", 42) == 42

    monkeypatch.setenv("DGM_TEST_VAR", "fast")
    try:
        module._env_positive_int("DGM_TEST_VAR", 42)
    except ValueError as exc:
        assert "DGM_TEST_VAR" in str(exc)
    else:
        raise AssertionError("expected ValueError on non-numeric input")

    monkeypatch.setenv("DGM_TEST_VAR", "0")
    try:
        module._env_positive_int("DGM_TEST_VAR", 42)
    except ValueError as exc:
        assert "DGM_TEST_VAR" in str(exc)
    else:
        raise AssertionError("expected ValueError on zero value")

    monkeypatch.setenv("DGM_TEST_VAR", "  900  ")
    assert module._env_positive_int("DGM_TEST_VAR", 42) == 900
