"""Tests for env-driven polyglot agent timeout resolution (kcsi #1125)."""

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
        "polyglot.leak_scrub",
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
    fake_leak_scrub = ModuleType("polyglot.leak_scrub")
    fake_leak_scrub.pre_agent_lockdown_script = lambda *_args, **_kwargs: ""
    fake_leak_scrub.build_grade_bundle = lambda *_args, **_kwargs: None
    fake_leak_scrub.inject_grade_bundle_script = lambda *_args, **_kwargs: ""

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
        sys.modules["polyglot.leak_scrub"] = fake_leak_scrub
        sys.modules["swe_bench"] = fake_swe_bench
        sys.modules["swe_bench.utils"] = fake_swe_utils
        sys.modules["utils"] = fake_utils
        sys.modules["utils.git_utils"] = fake_git_utils
        return _load_module("polyglot/harness.py", "dgm_polyglot_harness_timeout")
    finally:
        for module_name, original in originals.items():
            if original is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original


def test_default_is_600(monkeypatch):
    monkeypatch.delenv("CROSS_RUNNER_AGENT_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("DGM_POLYGLOT_AGENT_TIMEOUT_SEC", raising=False)
    module = _load_polyglot_harness()
    assert module._polyglot_agent_timeout_sec() == 600


def test_cross_runner_env_overrides_default(monkeypatch):
    monkeypatch.setenv("CROSS_RUNNER_AGENT_TIMEOUT_SEC", "3600")
    monkeypatch.delenv("DGM_POLYGLOT_AGENT_TIMEOUT_SEC", raising=False)
    module = _load_polyglot_harness()
    assert module._polyglot_agent_timeout_sec() == 3600


def test_dgm_specific_env_overrides_default(monkeypatch):
    monkeypatch.delenv("CROSS_RUNNER_AGENT_TIMEOUT_SEC", raising=False)
    monkeypatch.setenv("DGM_POLYGLOT_AGENT_TIMEOUT_SEC", "1800")
    module = _load_polyglot_harness()
    assert module._polyglot_agent_timeout_sec() == 1800


def test_cross_runner_wins_over_dgm_specific(monkeypatch):
    monkeypatch.setenv("CROSS_RUNNER_AGENT_TIMEOUT_SEC", "3600")
    monkeypatch.setenv("DGM_POLYGLOT_AGENT_TIMEOUT_SEC", "1800")
    module = _load_polyglot_harness()
    assert module._polyglot_agent_timeout_sec() == 3600


def test_cross_runner_zero_falls_through_to_default(monkeypatch):
    monkeypatch.setenv("CROSS_RUNNER_AGENT_TIMEOUT_SEC", "0")
    monkeypatch.delenv("DGM_POLYGLOT_AGENT_TIMEOUT_SEC", raising=False)
    module = _load_polyglot_harness()
    assert module._polyglot_agent_timeout_sec() == 600
