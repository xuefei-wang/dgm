import importlib.util
import sys
from pathlib import Path
from types import ModuleType


DGM_ROOT = Path(__file__).resolve().parents[1]


def _load_module(relative_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, DGM_ROOT / relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_polyglot_harness():
    module_names = [
        "docker",
        "dotenv",
        "prompts",
        "prompts.testrepo_prompt",
        "polyglot",
        "polyglot.test_spec",
        "polyglot.docker_build",
        "polyglot.constants",
        "swe_bench",
        "swe_bench.utils",
        "utils",
        "utils.git_utils",
    ]
    originals = {module_name: sys.modules.get(module_name) for module_name in module_names}
    for module_name in module_names:
        sys.modules.pop(module_name, None)

    fake_docker = ModuleType("docker")
    fake_dotenv = ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *_args, **_kwargs: None

    fake_prompts = ModuleType("prompts")
    fake_prompts_testrepo = ModuleType("prompts.testrepo_prompt")
    fake_prompts_testrepo.get_test_description = lambda *_args, **_kwargs: "test description"

    fake_polyglot = ModuleType("polyglot")
    fake_polyglot.__path__ = []
    fake_test_spec = ModuleType("polyglot.test_spec")
    fake_test_spec.make_test_spec = lambda *_args, **_kwargs: None
    fake_docker_build = ModuleType("polyglot.docker_build")
    fake_docker_build.build_env_images = lambda *_args, **_kwargs: None
    fake_docker_build.build_container = lambda *_args, **_kwargs: None
    fake_docker_build.cleanup_container = lambda *_args, **_kwargs: None
    fake_constants = ModuleType("polyglot.constants")
    fake_constants.MAP_REPO_VERSION_TO_SPECS = {}
    fake_constants.TEST_COMMANDS = {}

    fake_swe_bench = ModuleType("swe_bench")
    fake_swe_bench.__path__ = []
    fake_swe_utils = ModuleType("swe_bench.utils")
    for name in [
        "copy_to_container",
        "copy_from_container",
        "log_container_output",
        "remove_existing_container",
        "safe_log",
        "setup_logger",
    ]:
        setattr(fake_swe_utils, name, lambda *_args, **_kwargs: None)

    fake_utils = ModuleType("utils")
    fake_utils.__path__ = []
    fake_git_utils = ModuleType("utils.git_utils")
    fake_git_utils.filter_patch_by_files = lambda patch, *_args, **_kwargs: patch
    fake_git_utils.remove_patch_by_files = lambda patch, *_args, **_kwargs: patch

    try:
        sys.modules["docker"] = fake_docker
        sys.modules["dotenv"] = fake_dotenv
        sys.modules["prompts"] = fake_prompts
        sys.modules["prompts.testrepo_prompt"] = fake_prompts_testrepo
        sys.modules["polyglot"] = fake_polyglot
        sys.modules["polyglot.test_spec"] = fake_test_spec
        sys.modules["polyglot.docker_build"] = fake_docker_build
        sys.modules["polyglot.constants"] = fake_constants
        sys.modules["swe_bench"] = fake_swe_bench
        sys.modules["swe_bench.utils"] = fake_swe_utils
        sys.modules["utils"] = fake_utils
        sys.modules["utils.git_utils"] = fake_git_utils
        return _load_module("polyglot/harness.py", "dgm_polyglot_harness")
    finally:
        for module_name, original in originals.items():
            if original is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original


def test_polyglot_report_uses_selected_subset_denominator() -> None:
    module = _load_polyglot_harness()
    entries = [
        {"instance_id": "a"},
        {"instance_id": "b"},
        {"instance_id": "c"},
    ]
    results = [
        {"instance_id": "a", "success": True, "eval_result": "resolved"},
        {"instance_id": "b", "success": True, "eval_result": "unresolved"},
        {"instance_id": "c", "success": True, "eval_result": "empty_patch"},
    ]

    report = module.build_report(entries, results)

    assert report["total_instances"] == 3
    assert report["submitted_instances"] == 3
    assert report["resolved_ids"] == ["a"]
    assert report["unresolved_ids"] == ["b"]
    assert report["empty_patch_ids"] == ["c"]


def test_polyglot_initial_metadata_uses_task_map_denominator(tmp_path: Path) -> None:
    module = _load_module("polyglot/make_initial_metadata.py", "dgm_polyglot_make_initial_metadata")
    report = {
        "submitted_ids": ["a", "b", "c"],
        "resolved_ids": ["a"],
        "unresolved_ids": ["b"],
        "empty_patch_ids": ["c"],
        "incomplete_ids": [],
        "error_ids": [],
        "submitted_instances": 99,
    }

    metadata = module.build_metadata(report, ["a", "b", "c"], tmp_path / "report.json")

    overall = metadata["overall_performance"]
    assert overall["accuracy_score"] == 1 / 3
    assert overall["total_submitted_instances"] == 3
    assert overall["total_resolved_ids"] == ["a"]
    assert overall["total_unresolved_ids"] == ["b"]
    assert overall["total_emptypatch_ids"] == ["c"]


def test_polyglot_initial_metadata_rejects_task_map_mismatch(tmp_path: Path) -> None:
    module = _load_module(
        "polyglot/make_initial_metadata.py",
        "dgm_polyglot_make_initial_metadata_mismatch",
    )
    report = {
        "submitted_ids": ["a", "extra"],
        "resolved_ids": ["a"],
        "unresolved_ids": [],
        "empty_patch_ids": [],
        "incomplete_ids": [],
        "error_ids": [],
        "submitted_instances": 2,
    }

    try:
        module.build_metadata(report, ["a", "b"], tmp_path / "report.json")
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected task-map/report mismatch to fail")

    assert "missing=['b']" in message
    assert "extra=['extra']" in message
