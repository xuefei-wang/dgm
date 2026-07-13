from prompts import diagnose_improvement_prompt as diagnose_prompt


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
