import random

import pytest

from prompts import diagnose_improvement_prompt as diagnose_prompt
from prompts import self_improvement_prompt as self_prompt


def _patch_diagnosis_dependencies(monkeypatch):
    def fake_find(entry_id, out_dir, commit_id="initial", filter=True):
        return (
            [f"agent log {commit_id}"],
            [f"SECRET_EVAL_LOG_{commit_id}"],
            [f"predicted patch {commit_id}"],
            ["resolved" if commit_id == "child" else "unresolved"],
        )

    monkeypatch.setattr(diagnose_prompt, "find_selfimprove_eval_logs", fake_find)
    monkeypatch.setattr(diagnose_prompt, "get_current_code", lambda *args, **kwargs: "agent code")
    monkeypatch.setattr(diagnose_prompt, "read_file", lambda path: "model patch")


def _patch_self_prompt_dependencies(monkeypatch, eval_result):
    monkeypatch.setattr(
        self_prompt,
        "find_selfimprove_eval_logs",
        lambda entry_id, out_dir, commit_id="initial", filter=True: (
            ["agent log"],
            ["SECRET_EVAL_LOG"],
            ["predicted patch"],
            [eval_result],
        ),
    )
    monkeypatch.setattr(
        self_prompt,
        "process_selfimprove_eval_logs",
        lambda md_logs, eval_logs, predicted_patches, eval_results: (
            md_logs[0],
            eval_logs[0],
            predicted_patches[0],
            eval_results[0],
        ),
    )
    monkeypatch.setattr(self_prompt, "get_current_code", lambda *args, **kwargs: "agent code")


@pytest.mark.parametrize(
    ("eval_result", "expected"),
    [
        (None, "unresolved (missing eval result)"),
        ("eval_timeout", "unresolved (eval timeout)"),
        ("pending_eval", "unresolved (pending eval)"),
        ("eval_error: docker failed", "unresolved (eval error)"),
        ("empty_patch", "unresolved (empty patch)"),
        ("resolved", "resolved"),
    ],
)
def test_scalar_outcome_preserves_eval_status_categories(eval_result, expected):
    assert expected in self_prompt._matched_regime_scalar_outcome(eval_result)


def test_direct_swe_diagnosis_withholds_hidden_material_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("KCSI_DGM_LEAK_HIDDEN_MATERIAL", raising=False)
    _patch_self_prompt_dependencies(monkeypatch, "eval_timeout")

    system_prompt, user_prompt = self_prompt.get_diagnose_prompt_swe(
        "task-1",
        "parent",
        str(tmp_path),
        str(tmp_path / "out"),
        [
            {
                "instance_id": "task-1",
                "patch": "SECRET_GOLD_PATCH",
                "test_patch": "SECRET_TEST_PATCH",
                "problem_statement": "issue text",
            }
        ],
    )

    rendered = system_prompt + "\n" + user_prompt
    assert "SECRET_GOLD_PATCH" not in rendered
    assert "SECRET_TEST_PATCH" not in rendered
    assert "SECRET_EVAL_LOG" not in rendered
    assert "Scalar outcome only: unresolved (eval timeout)" in rendered


def test_direct_polyglot_diagnosis_withholds_hidden_material_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("KCSI_DGM_LEAK_HIDDEN_MATERIAL", raising=False)
    monkeypatch.setattr(random, "random", lambda: 1.0)
    _patch_self_prompt_dependencies(monkeypatch, "eval_error: tests failed")

    system_prompt, user_prompt = self_prompt.get_diagnose_prompt_polyglot(
        "poly-1",
        "parent",
        str(tmp_path),
        str(tmp_path / "out"),
        [
            {
                "instance_id": "poly-1",
                "language": "python",
                "reference_answers": "SECRET_REFERENCE_ANSWER",
                "reference_tests": "SECRET_REFERENCE_TEST",
                "problem_statement": "exercise text",
            }
        ],
    )

    rendered = system_prompt + "\n" + user_prompt
    assert "SECRET_REFERENCE_ANSWER" not in rendered
    assert "SECRET_REFERENCE_TEST" not in rendered
    assert "SECRET_EVAL_LOG" not in rendered
    assert "Scalar outcome only: unresolved (eval error)" in rendered


def test_direct_diagnosis_missing_dataset_entry_raises_clear_error(monkeypatch, tmp_path):
    _patch_self_prompt_dependencies(monkeypatch, "resolved")

    with pytest.raises(ValueError, match="Could not find entry"):
        self_prompt.get_diagnose_prompt_swe(
            "missing-task",
            "parent",
            str(tmp_path),
            str(tmp_path / "out"),
            [],
        )


def test_post_improvement_diagnosis_withholds_hidden_material_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("KCSI_DGM_LEAK_HIDDEN_MATERIAL", raising=False)
    _patch_diagnosis_dependencies(monkeypatch)

    system_prompt, user_prompt = diagnose_prompt.get_diagnose_improvement_prompt(
        "task-1",
        "parent",
        str(tmp_path),
        str(tmp_path / "model.patch"),
        str(tmp_path / "out"),
        "child",
        [{"instance_id": "task-1", "patch": "SECRET_GOLD_PATCH", "test_patch": "SECRET_TEST_PATCH"}],
    )

    rendered = system_prompt + "\n" + user_prompt
    assert "SECRET_GOLD_PATCH" not in rendered
    assert "SECRET_TEST_PATCH" not in rendered
    assert "SECRET_EVAL_LOG_parent" not in rendered
    assert "SECRET_EVAL_LOG_child" not in rendered
    assert "Scalar outcome only: unresolved" in rendered
    assert "Scalar outcome only: resolved" in rendered


def test_post_improvement_diagnosis_leak_flag_restores_upstream_material(monkeypatch, tmp_path):
    monkeypatch.setenv("KCSI_DGM_LEAK_HIDDEN_MATERIAL", "1")
    _patch_diagnosis_dependencies(monkeypatch)

    system_prompt, user_prompt = diagnose_prompt.get_diagnose_improvement_prompt(
        "task-1",
        "parent",
        str(tmp_path),
        str(tmp_path / "model.patch"),
        str(tmp_path / "out"),
        "child",
        [{"instance_id": "task-1", "patch": "SECRET_GOLD_PATCH", "test_patch": "SECRET_TEST_PATCH"}],
    )

    rendered = system_prompt + "\n" + user_prompt
    assert "SECRET_GOLD_PATCH" in rendered
    assert "SECRET_TEST_PATCH" in rendered
    assert "SECRET_EVAL_LOG_parent" in rendered
    assert "SECRET_EVAL_LOG_child" in rendered
