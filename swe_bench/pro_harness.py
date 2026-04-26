#!/usr/bin/env python3
"""DGM harness for SWE-bench Pro tasks."""

from __future__ import annotations

import argparse
import ast
import datetime
import json
import os
import posixpath
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional when only using helper functions
    load_dotenv = None


DGM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(DGM_ROOT) not in sys.path:
    sys.path.insert(0, str(DGM_ROOT))
DEFAULT_DATASET_PATH = REPO_ROOT / "benchmarks" / "swebench_pro" / "dataset" / "test.jsonl"
DEFAULT_TASK_MAP = REPO_ROOT / "benchmarks" / "swebench_pro" / "task_maps" / "swebench_pro_test_50_seed0_v1.json"
DEFAULT_EVAL_SOURCE = REPO_ROOT / "third_party" / "SWE-bench_Pro-os"
DEFAULT_SCRIPTS_DIR = DEFAULT_EVAL_SOURCE / "run_scripts"
DEFAULT_DOCKERHUB_USERNAME = "jefzda"


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        raise ValueError(f"Env var {name} must be a positive integer, got {raw!r}")
    if value <= 0:
        raise ValueError(f"Env var {name} must be > 0, got {value!r}")
    return value


DEFAULT_OFFICIAL_EVAL_TIMEOUT_SEC = _env_positive_int("DGM_SWEBENCH_OFFICIAL_EVAL_TIMEOUT_SEC", 3600)
# Match upstream DGM's SWE-bench Verified harness, which hardcodes
# `timeout 32400` (9h) on the agent. Changing this would shorten DGM's
# per-task budget on Pro vs. published Verified, biasing the comparison.
# Configurable via env var so devs can tune for iteration; production
# campaigns should leave it unset.
DEFAULT_AGENT_TIMEOUT_SEC = _env_positive_int("DGM_SWEBENCH_AGENT_TIMEOUT_SEC", 32400)
AGENT_PIP_INDEX_URL = "https://pypi.org/simple"
SAFE_INSTANCE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def validate_instance_id(instance_id: Any) -> str:
    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError(f"Invalid SWE-bench Pro instance ID: {instance_id!r}")
    if instance_id in {".", ".."} or not SAFE_INSTANCE_ID_RE.fullmatch(instance_id):
        raise ValueError(f"Unsafe SWE-bench Pro instance ID: {instance_id!r}")
    return instance_id


def safe_instance_filename(instance_id: Any) -> str:
    return validate_instance_id(instance_id)


def _load_shared_env() -> None:
    if load_dotenv is None:
        return
    for env_path in [
        REPO_ROOT / "configs" / "providers" / ".env.shared",
        REPO_ROOT / "configs" / "providers" / ".env.haiku",
        REPO_ROOT / "configs" / "providers" / ".env.openai",
        REPO_ROOT / "configs" / "models" / "shared.env",
    ]:
        if env_path.exists():
            load_dotenv(env_path, override=True)


def _collect_runtime_env(names: list[str]) -> dict[str, str]:
    env_vars = {}
    for name in names:
        value = os.getenv(name)
        if value:
            env_vars[name] = value
    return env_vars


def _runtime_env() -> dict[str, str]:
    return _collect_runtime_env(
        [
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_BASE_URL",
            "ANTHROPIC_BEDROCK_BASE_URL",
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "OPENAI_ORG_ID",
            "OPENAI_PROJECT_ID",
            "GEMINI_API_KEY",
            "OPENROUTER_API_KEY",
            "DEEPSEEK_API_KEY",
            "AWS_REGION",
            "AWS_REGION_NAME",
            "AWS_DEFAULT_REGION",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "DGM_CLAUDE_MODEL",
            "DGM_OPENAI_MODEL",
            "DGM_CODE_MODEL",
            "DGM_SELF_IMPROVE_MODEL",
            "DGM_DIAGNOSE_MODEL",
            "DGM_REASONING_EFFORT",
            "OPENAI_REASONING_EFFORT",
            "REASONING_EFFORT",
        ]
    )


def _agent_pip_install_command(python_bin: str, *, break_system_packages: bool) -> str:
    cmd = [
        "env",
        "-u",
        "PIP_INDEX_URL",
        "-u",
        "PIP_EXTRA_INDEX_URL",
        "PIP_CONFIG_FILE=/dev/null",
        "PIP_DISABLE_PIP_VERSION_CHECK=1",
        python_bin,
        "-m",
        "pip",
        "install",
        "--isolated",
        "--index-url",
        AGENT_PIP_INDEX_URL,
    ]
    if break_system_packages:
        cmd.append("--break-system-packages")
    cmd.extend(["-r", "/dgm/requirements.txt"])
    return shlex.join(cmd)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    entries = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if payload:
                row = json.loads(payload)
                if isinstance(row, dict):
                    entries.append(row)
    return entries


def load_task_ids(path: Path | str | None) -> list[str] | None:
    if path is None:
        return None
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list) and all(isinstance(item, str) for item in data):
        return [validate_instance_id(item) for item in data]
    if isinstance(data, dict):
        if isinstance(data.get("task_ids"), list):
            return [validate_instance_id(item) for item in data["task_ids"] if isinstance(item, str)]
        if isinstance(data.get("tasks"), list):
            ids = []
            for item in data["tasks"]:
                if not isinstance(item, dict):
                    continue
                task_id = item.get("task_id") or item.get("instance_id")
                if isinstance(task_id, str):
                    ids.append(validate_instance_id(task_id))
            return ids
    raise ValueError(f"Unsupported task map format: {path}")


def _select_entries(entries: list[dict[str, Any]], task_ids: list[str] | None) -> list[dict[str, Any]]:
    if task_ids is None:
        return [{**entry, "instance_id": validate_instance_id(entry.get("instance_id"))} for entry in entries]
    task_ids = [validate_instance_id(task_id) for task_id in task_ids]
    by_id = {validate_instance_id(entry.get("instance_id")): entry for entry in entries}
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} task IDs missing from dataset: {missing[:10]}")
    return [{**by_id[task_id], "instance_id": task_id} for task_id in task_ids]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) >= 2 and text[0] in {"'", '"'} and text[-1] == text[0]:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return text
        if isinstance(parsed, str):
            return parsed.strip()
    return text


def _parse_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return [text]
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return []


def _normalize_repo_path(path: str) -> str:
    path = path.strip()
    if not path or path == "/dev/null":
        return ""
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    while path.startswith("./"):
        path = path[2:]
    path = path.lstrip("/")
    normalized = posixpath.normpath(path)
    if normalized in {".", ".."} or normalized.startswith("../"):
        return ""
    return normalized


def _private_test_path_candidates(value: str) -> list[str]:
    candidates = []
    text = value.strip()
    if not text:
        return candidates
    candidates.append(text)
    if " | " in text:
        candidates.append(text.split(" | ", 1)[0])
    if "::" in text:
        candidates.append(text.split("::", 1)[0])
    return candidates


def _private_test_paths(entry: dict[str, Any]) -> set[str]:
    paths = set()
    for field in [
        "selected_test_files_to_run",
        "fail_to_pass",
        "FAIL_TO_PASS",
        "pass_to_pass",
        "PASS_TO_PASS",
    ]:
        for value in _parse_string_list(entry.get(field)):
            for candidate in _private_test_path_candidates(value):
                normalized = _normalize_repo_path(candidate)
                if normalized:
                    paths.add(normalized)
    return paths


def _diff_header_path(value: str) -> str:
    value = value.strip()
    if "\t" in value:
        value = value.split("\t", 1)[0]
    return _normalize_repo_path(value)


def _diff_block_paths(block: list[str]) -> set[str]:
    paths = set()
    for line in block:
        if line.startswith("diff --git "):
            try:
                parts = shlex.split(line.strip())
            except ValueError:
                parts = line.strip().split()
            if len(parts) >= 4:
                for value in parts[2:4]:
                    normalized = _normalize_repo_path(value)
                    if normalized:
                        paths.add(normalized)
        elif line.startswith("--- ") or line.startswith("+++ "):
            normalized = _diff_header_path(line[4:])
            if normalized:
                paths.add(normalized)
    return paths


def filter_private_test_patch_hunks(patch: str, entry: dict[str, Any]) -> str:
    private_paths = _private_test_paths(entry)
    if not private_paths or not patch.strip():
        return patch

    blocks = []
    current: list[str] = []
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git ") and current:
            blocks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(current)

    filtered_blocks = []
    for block in blocks:
        if _diff_block_paths(block) & private_paths:
            continue
        filtered_blocks.extend(block)
    return "".join(filtered_blocks)


def _build_problem_statement(entry: dict[str, Any]) -> str:
    parts = [_clean_text(entry.get("problem_statement"))]
    requirements = _clean_text(entry.get("requirements"))
    interface = _clean_text(entry.get("interface"))
    if requirements:
        parts.extend(["", "Requirements:", requirements])
    if interface:
        parts.extend(["", "Relevant interface:", interface])
    return "\n".join(part for part in parts if part is not None).strip()


def _build_test_description(entry: dict[str, Any]) -> str:
    selected_files = _parse_string_list(entry.get("selected_test_files_to_run"))
    lines = [
        "SWE-bench Pro evaluates this repository with the official per-instance Docker image.",
        "When the official test script is available in this container, run it with:",
        "`cd /app && bash /workspace/run_script.sh <specific test files>`.",
        "Use the given command shape exactly; omit <specific test files> to run the full script.",
    ]
    if selected_files:
        lines.append("Selected test files for this issue:")
        lines.extend(f"- {path}" for path in selected_files)
    lines.append("Do not solve the issue by modifying tests; make the minimal source change.")
    return "\n".join(lines)


def _last_before_repo_set_cmd(entry: dict[str, Any]) -> str:
    commands = _clean_text(entry.get("before_repo_set_cmd")).splitlines()
    commands = [command.strip() for command in commands if command.strip()]
    return commands[-1] if commands else ""


def _safe_container_name(instance_id: str, run_id: str) -> str:
    instance_id = validate_instance_id(instance_id)
    value = re.sub(r"[^a-zA-Z0-9_.-]", "-", f"dgm-pro-{instance_id}-{run_id}")
    return value[:120].strip("-.") or f"dgm-pro-{run_id}"


def _dockerhub_image_uri(entry: dict[str, Any], dockerhub_username: str) -> str:
    uid = validate_instance_id(entry["instance_id"])
    repo_name = str(entry.get("repo") or "")
    repo_base, repo_name_only = repo_name.lower().split("/")
    hsh = uid.replace("instance_", "")

    if uid == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan":
        repo_name_only = "element-web"
    elif "element-hq" in repo_name.lower() and "element-web" in repo_name.lower():
        repo_name_only = "element"
        if hsh.endswith("-vnan"):
            hsh = hsh[:-5]
    elif hsh.endswith("-vnan"):
        hsh = hsh[:-5]

    tag = f"{repo_base}.{repo_name_only}-{hsh}"
    if len(tag) > 128:
        tag = tag[:128]
    return f"{dockerhub_username}/sweap-images:{tag}"


def _container_python(container) -> str:
    probe = "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
    for candidate in ("python3", "python"):
        result = container.exec_run([candidate, "-c", probe], workdir="/")
        if result.exit_code == 0 and result.output.decode("utf-8", errors="ignore").startswith("3."):
            return candidate
    raise RuntimeError("Could not find a usable Python 3 interpreter in the SWE-bench Pro container")


def _setup_agent_python(container, python_bin: str) -> str:
    from swe_bench.utils import log_container_output

    venv_install_cmd = _agent_pip_install_command("/dgm/.venv/bin/python", break_system_packages=False)
    system_install_cmd = _agent_pip_install_command(python_bin, break_system_packages=True)

    result = container.exec_run([python_bin, "-m", "venv", "/dgm/.venv"], workdir="/")
    log_container_output(result, raise_error=False)
    if result.exit_code == 0:
        result = container.exec_run(["/bin/bash", "-lc", venv_install_cmd], workdir="/")
        log_container_output(result, raise_error=False)
        if result.exit_code == 0:
            return "/dgm/.venv/bin/python"
        log_container_output(container.exec_run(["rm", "-rf", "/dgm/.venv"], workdir="/"), raise_error=False)

    result = container.exec_run(["/bin/bash", "-lc", system_install_cmd], workdir="/")
    log_container_output(result)
    return python_bin


def _copy_dgm_runtime(container, scripts_dir: Path, instance_id: str) -> None:
    from swe_bench.utils import copy_to_container

    instance_dirname = safe_instance_filename(instance_id)
    container.exec_run("mkdir -p /dgm /workspace", workdir="/")
    for relative in [
        "coding_agent.py",
        "requirements.txt",
        "pytest.ini",
        "tools",
        "utils",
        "tests",
        "prompts",
        "llm.py",
        "llm_withtools.py",
    ]:
        source = DGM_ROOT / relative
        dest = f"/dgm/{relative}"
        copy_to_container(container, source, dest)

    run_script = scripts_dir / instance_dirname / "run_script.sh"
    parser_script = scripts_dir / instance_dirname / "parser.py"
    if run_script.exists():
        copy_to_container(container, run_script, "/workspace/run_script.sh")
        container.exec_run("chmod +x /workspace/run_script.sh", workdir="/")
    if parser_script.exists():
        copy_to_container(container, parser_script, "/workspace/parser.py")


def _prepare_app_repo(container, entry: dict[str, Any]) -> str:
    from swe_bench.utils import log_container_output

    base_commit = str(entry["base_commit"])
    setup = (
        "git config --global --add safe.directory /app || true && "
        f"git -C /app reset --hard {base_commit} && "
        f"git -C /app checkout {base_commit} && "
        "git -C /app clean -fd"
    )
    log_container_output(container.exec_run(["/bin/bash", "-lc", setup], workdir="/"))

    before_cmd = _last_before_repo_set_cmd(entry)
    if before_cmd:
        log_container_output(container.exec_run(["/bin/bash", "-lc", before_cmd], workdir="/app"))

    commit_cmd = (
        "git -C /app add --all && "
        "git -C /app -c user.name='dgm' -c user.email='dgm@example.com' "
        "commit -m 'swebench pro agent baseline' >/tmp/dgm-pro-baseline.out 2>&1 || true && "
        "git -C /app rev-parse HEAD"
    )
    result = container.exec_run(["/bin/bash", "-lc", commit_cmd], workdir="/")
    log_container_output(result)
    return result.output.decode("utf-8").strip().splitlines()[-1]


def _apply_model_patches(container, model_patch_paths: list[str] | None) -> None:
    if not model_patch_paths:
        return
    from swe_bench.utils import copy_to_container, log_container_output, safe_log

    safe_log("Applying DGM model patches")
    for model_patch_path in model_patch_paths:
        copy_to_container(container, model_patch_path, "/dgm/parent_patch.txt")
        log_container_output(container.exec_run("/bin/sh -c 'patch -p1 < /dgm/parent_patch.txt'", workdir="/dgm"))
        log_container_output(container.exec_run("rm /dgm/parent_patch.txt", workdir="/dgm"))


def process_entry(
    entry: dict[str, Any],
    out_dname: Path,
    model_name_or_path: str,
    model_patch_paths: list[str] | None,
    *,
    scripts_dir: Path,
    dockerhub_username: str,
    docker_platform: str | None = None,
) -> dict[str, Any]:
    instance_id = validate_instance_id(entry["instance_id"])
    instance_filename = safe_instance_filename(instance_id)
    chat_history_file = out_dname / f"{instance_filename}.md"
    out_fname = out_dname / f"{instance_filename}.json"

    if out_fname.exists():
        with out_fname.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        resumed = {
            "success": bool(existing.get("success", True)),
            "instance_id": instance_id,
            "model_patch": existing.get("model_patch", ""),
            "json_path": str(out_fname),
        }
        # Propagate agent-timeout signal so _write_eval_logs reaches the same
        # branch on resume that it would on a fresh run.
        if "agent_status" in existing:
            resumed["agent_status"] = existing["agent_status"]
        if "agent_exit_code" in existing:
            resumed["agent_exit_code"] = existing["agent_exit_code"]
        return resumed

    client = None
    container = None
    model_patch = ""
    proposed_model_patches: list[str] = []
    try:
        _load_shared_env()
        import docker
        from swe_bench.utils import copy_from_container, log_container_output, remove_existing_container, setup_logger

        client = docker.from_env()
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        setup_logger(str(out_dname / f"{instance_filename}_docker.log"))
        image_uri = _dockerhub_image_uri(entry, dockerhub_username)
        try:
            if docker_platform:
                client.images.pull(image_uri, platform=docker_platform)
            else:
                client.images.pull(image_uri)
        except Exception:
            client.images.get(image_uri)

        container_name = _safe_container_name(instance_id, run_id)
        remove_existing_container(client, container_name)
        run_kwargs: dict[str, Any] = {
            "name": container_name,
            "detach": True,
            "entrypoint": "/bin/bash",
            "command": ["-c", "tail -f /dev/null"],
        }
        if docker_platform:
            run_kwargs["platform"] = docker_platform
        container = client.containers.run(image_uri, **run_kwargs)

        _copy_dgm_runtime(container, scripts_dir, instance_id)
        agent_base_commit = _prepare_app_repo(container, entry)
        _apply_model_patches(container, model_patch_paths)

        python_bin = _container_python(container)
        agent_python = _setup_agent_python(container, python_bin)

        chat_history_file_container = f"/dgm/{chat_history_file.name}"
        cmd = [
            "timeout",
            str(max(1, DEFAULT_AGENT_TIMEOUT_SEC)),
            agent_python,
            "/dgm/coding_agent.py",
            "--problem_statement",
            _build_problem_statement(entry),
            "--git_dir",
            "/app/",
            "--chat_history_file",
            chat_history_file_container,
            "--base_commit",
            agent_base_commit,
            "--outdir",
            "/dgm/",
            "--test_description",
            _build_test_description(entry),
            "--instance_id",
            instance_id,
        ]
        agent_result = container.exec_run(cmd, environment=_runtime_env(), workdir="/app")
        log_container_output(agent_result, raise_error=False)

        copy_from_container(container, chat_history_file_container, chat_history_file)
        result = container.exec_run(["find", "/dgm/", "-name", f"{instance_filename}_*.md"], workdir="/")
        for history_path in result.output.decode("utf-8").split():
            copy_from_container(container, history_path, out_dname / Path(history_path).name)

        patch_result = container.exec_run("cat /dgm/model_patch.diff", workdir="/")
        model_patch = patch_result.output.decode("utf-8") if patch_result.exit_code == 0 else ""

        result = container.exec_run("find /dgm/ -name 'model_patch_*.diff'", workdir="/")
        for patch_path in result.output.decode("utf-8").split():
            patch_content = container.exec_run(f"cat {patch_path}", workdir="/")
            if patch_content.exit_code == 0:
                proposed_model_patches.append(patch_content.output.decode("utf-8"))

        agent_exit_code = int(getattr(agent_result, "exit_code", 0) or 0)
        agent_status = "timeout" if agent_exit_code == 124 else "ok"
        prediction = {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": model_patch,
            "proposed_model_patches": proposed_model_patches,
            "agent_status": agent_status,
            "agent_exit_code": agent_exit_code,
            "eval_result": "pending_eval" if model_patch.strip() else ("agent_timeout" if agent_status == "timeout" else "empty_patch"),
            "success": True,
        }
        out_fname.write_text(json.dumps(prediction, indent=4), encoding="utf-8")
        return {
            "success": True,
            "instance_id": instance_id,
            "model_patch": model_patch,
            "agent_status": agent_status,
            "agent_exit_code": agent_exit_code,
            "json_path": str(out_fname),
        }
    except Exception as exc:
        prediction = {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": model_patch,
            "proposed_model_patches": proposed_model_patches,
            "eval_result": "error",
            "success": False,
            "error": str(exc),
        }
        out_fname.write_text(json.dumps(prediction, indent=4), encoding="utf-8")
        print(f"Error processing entry {instance_id}: {exc}")
        return {
            "success": False,
            "instance_id": instance_id,
            "model_patch": model_patch,
            "json_path": str(out_fname),
            "error": str(exc),
        }
    finally:
        if container is not None:
            try:
                container.stop()
                container.remove()
            except Exception as exc:
                print(f"Error cleaning up Docker container for {instance_id}: {exc}")


def _write_patch_bundle(
    path: Path,
    results: list[dict[str, Any]],
    prefix: str,
    entries_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    payload = []
    for result in results:
        patch = str(result.get("model_patch") or "")
        if not result.get("success") or not patch.strip():
            continue
        instance_id = validate_instance_id(result["instance_id"])
        patch = filter_private_test_patch_hunks(patch, entries_by_id[instance_id])
        result["eval_patch"] = patch
        if not patch.strip():
            continue
        payload.append(
            {
                "instance_id": instance_id,
                "patch": patch,
                "prefix": prefix,
            }
        )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _run_official_eval(
    *,
    patch_bundle: Path,
    official_eval_dir: Path,
    dataset_path: Path,
    eval_source: Path,
    scripts_dir: Path,
    dockerhub_username: str,
    max_workers: int,
    use_local_docker: bool,
    docker_platform: str | None,
    block_network: bool,
    timeout_sec: int = DEFAULT_OFFICIAL_EVAL_TIMEOUT_SEC,
) -> dict[str, Any]:
    official_eval_dir.mkdir(parents=True, exist_ok=True)
    wrapper = REPO_ROOT / "scripts" / "run_swebench_pro_eval.py"
    if wrapper.exists():
        cmd = [
            sys.executable,
            str(wrapper),
            "--patch-path",
            str(patch_bundle),
            "--output-dir",
            str(official_eval_dir),
            "--source-dir",
            str(eval_source),
            "--raw-sample-path",
            str(dataset_path),
            "--scripts-dir",
            str(scripts_dir),
            "--dockerhub-username",
            dockerhub_username,
            "--num-workers",
            str(max(1, max_workers)),
            "--redo",
        ]
        if use_local_docker:
            cmd.append("--use-local-docker")
        if docker_platform:
            cmd.extend(["--docker-platform", docker_platform])
        if block_network:
            cmd.append("--block-network")
    else:
        cmd = [
            sys.executable,
            str(eval_source / "swe_bench_pro_eval.py"),
            "--raw_sample_path",
            str(dataset_path),
            "--patch_path",
            str(patch_bundle),
            "--output_dir",
            str(official_eval_dir),
            "--dockerhub_username",
            dockerhub_username,
            "--scripts_dir",
            str(scripts_dir),
            "--num_workers",
            str(max(1, max_workers)),
            "--redo",
        ]
        if use_local_docker:
            cmd.append("--use_local_docker")
        if docker_platform:
            cmd.extend(["--docker_platform", docker_platform])
        if block_network:
            cmd.append("--block_network")
    timeout_sec = max(1, int(timeout_sec))

    # Snapshot containers running before we start so we can identify (and
    # stop) only the ones this eval invocation spawned if it times out or
    # fails. The official evaluator launches up to num_workers detached
    # containers via the docker SDK; subprocess.run's SIGKILL on
    # TimeoutExpired kills our wrapper but leaves dockerd-managed containers
    # running, so we have to clean them up explicitly.
    pre_eval_container_ids = _snapshot_running_container_ids()

    leaked_stopped = 0
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        leaked_stopped = _stop_new_evaluator_containers(
            dockerhub_username=dockerhub_username,
            preexisting_ids=pre_eval_container_ids,
        )
        return {
            "status": "timeout",
            "timeout_sec": timeout_sec,
            "returncode": None,
            "stdout_tail": (exc.stdout or "")[-2000:],
            "stderr_tail": (exc.stderr or "")[-2000:],
            "leaked_containers_stopped": leaked_stopped,
        }

    if proc.returncode != 0:
        # Non-zero exit may also leave containers behind if the evaluator
        # crashed mid-run; same cleanup strategy applies.
        leaked_stopped = _stop_new_evaluator_containers(
            dockerhub_username=dockerhub_username,
            preexisting_ids=pre_eval_container_ids,
        )

    return {
        "status": "ok" if proc.returncode == 0 else "failed",
        "timeout_sec": timeout_sec,
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
        "leaked_containers_stopped": leaked_stopped,
    }


def _snapshot_running_container_ids() -> set[str]:
    """Best-effort snapshot of currently-running docker container IDs.
    Returns an empty set if the docker SDK is unavailable or the daemon is
    unreachable."""
    try:
        import docker
    except Exception:
        return set()
    try:
        client = docker.from_env()
        return {c.id for c in client.containers.list() if getattr(c, "id", None)}
    except Exception:
        return set()


def _stop_new_evaluator_containers(
    *,
    dockerhub_username: str,
    preexisting_ids: set[str],
) -> int:
    """Stop any container that (a) is not in `preexisting_ids` and (b) was
    launched from an image under `dockerhub_username/*`. Best-effort:
    swallow all errors (docker daemon unavailable, container already gone,
    image inspection failed) and return the number of containers we
    successfully stopped."""
    try:
        import docker
    except Exception:
        return 0
    try:
        client = docker.from_env()
    except Exception:
        return 0
    image_prefix = f"{dockerhub_username}/"
    stopped = 0
    try:
        containers = client.containers.list()
    except Exception:
        return 0
    for container in containers:
        cid = getattr(container, "id", None)
        if not cid or cid in preexisting_ids:
            continue
        try:
            tags = container.image.tags or []
        except Exception:
            continue
        if not any(tag.startswith(image_prefix) for tag in tags):
            continue
        try:
            container.stop(timeout=10)
            stopped += 1
        except Exception:
            continue
    return stopped


def _load_eval_results(official_eval_dir: Path) -> dict[str, bool]:
    eval_results_path = official_eval_dir / "eval_results.json"
    if not eval_results_path.exists():
        return {}
    data = json.loads(eval_results_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return {validate_instance_id(key): bool(value) for key, value in data.items()}


def _build_tests_status(output: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    tests = output.get("tests", []) if isinstance(output, dict) else []
    observed_tests = {}
    skipped_statuses = {"SKIPPED", "XFAIL", "ERROR"}
    for item in tests:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        status = str(item.get("status") or "").strip().upper()
        if name and status:
            observed_tests[name] = status

    def classify(expected: list[str]) -> dict[str, list[str]]:
        return {
            "success": [name for name in expected if observed_tests.get(name) == "PASSED"],
            "failure": [name for name in expected if observed_tests.get(name) == "FAILED"],
            "skipped": [name for name in expected if observed_tests.get(name) in skipped_statuses],
            "unknown": [name for name in expected if name not in observed_tests],
        }

    return {
        "observed_count": len(observed_tests),
        "FAIL_TO_PASS": classify(_parse_string_list(entry.get("fail_to_pass") or entry.get("FAIL_TO_PASS"))),
        "PASS_TO_PASS": classify(_parse_string_list(entry.get("pass_to_pass") or entry.get("PASS_TO_PASS"))),
    }


def _is_resolved_from_output(output: dict[str, Any], entry: dict[str, Any]) -> bool:
    tests = output.get("tests", []) if isinstance(output, dict) else []
    passed_tests = {
        str(item.get("name") or "").strip()
        for item in tests
        if isinstance(item, dict) and str(item.get("status") or "").strip().upper() == "PASSED"
    }
    expected = set(_parse_string_list(entry.get("fail_to_pass") or entry.get("FAIL_TO_PASS")))
    expected.update(_parse_string_list(entry.get("pass_to_pass") or entry.get("PASS_TO_PASS")))
    return expected <= passed_tests


def _load_partial_eval_results(
    official_eval_dir: Path,
    *,
    prefix: str,
    entries_by_id: dict[str, dict[str, Any]],
) -> dict[str, bool]:
    partial_results: dict[str, bool] = {}
    for instance_id, entry in entries_by_id.items():
        output_path = official_eval_dir / instance_id / f"{prefix}_output.json"
        if not output_path.exists():
            continue
        try:
            loaded = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(loaded, dict):
            continue
        partial_results[instance_id] = _is_resolved_from_output(loaded, entry)
    return partial_results


def _eval_log_text(resolved: bool | None, tests_status: dict[str, Any]) -> str:
    lines = [f"Resolved: {resolved}" if resolved is not None else "Resolved: unknown"]
    sections = [
        ("New tests for the issue", "FAIL_TO_PASS"),
        ("Previous tests from the repo", "PASS_TO_PASS"),
    ]
    for title, key in sections:
        lines.extend(["", f"## {title}"])
        status = tests_status.get(key, {})
        for label in ["success", "failure", "skipped", "unknown"]:
            values = status.get(label) or []
            if values:
                lines.append(f"{label}: {len(values)}")
                lines.extend(f"- {value}" for value in values)
        if not any(status.get(label) for label in ["success", "failure", "skipped", "unknown"]):
            lines.append("No parsed tests in this category.")
    return "\n".join(lines)


def _update_prediction(path: Path, *, eval_result: str) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["eval_result"] = eval_result
    # Atomic write: prediction JSON is also the resume checkpoint; a SIGKILL
    # mid-write would corrupt it and break subsequent process_entry resumes.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=4), encoding="utf-8")
    os.replace(tmp, path)


def _write_eval_logs(
    *,
    entries_by_id: dict[str, dict[str, Any]],
    results: list[dict[str, Any]],
    out_dname: Path,
    official_eval_dir: Path,
    prefix: str,
    eval_results: dict[str, bool],
    eval_metadata: dict[str, Any] | None = None,
) -> None:
    eval_metadata = eval_metadata or {}
    for result in results:
        instance_id = validate_instance_id(result["instance_id"])
        instance_filename = safe_instance_filename(instance_id)
        json_path = Path(str(result["json_path"]))
        if not result.get("success"):
            _update_prediction(json_path, eval_result="error")
            (out_dname / f"{instance_filename}_eval.md").write_text(
                f"Evaluation did not run because prediction generation failed.\n\n{result.get('error', '')}",
                encoding="utf-8",
            )
            continue
        if not str(result.get("eval_patch", result.get("model_patch")) or "").strip():
            if result.get("agent_status") == "timeout":
                _update_prediction(json_path, eval_result="agent_timeout")
                eval_text = (
                    "Evaluation did not run because prediction generation timed out "
                    f"after {DEFAULT_AGENT_TIMEOUT_SEC} seconds and no model patch was emitted."
                )
            else:
                _update_prediction(json_path, eval_result="empty_patch")
                eval_text = "Evaluation did not run because the model patch was empty."
            (out_dname / f"{instance_filename}_eval.md").write_text(eval_text, encoding="utf-8")
            continue

        resolved = eval_results.get(instance_id)
        output_path = official_eval_dir / instance_id / f"{prefix}_output.json"
        if resolved is None and eval_metadata.get("status") == "timeout":
            eval_result = "eval_timeout"
        elif resolved is None and eval_metadata.get("status") == "failed":
            eval_result = "eval_error"
        elif resolved is None and not output_path.exists():
            eval_result = "eval_incomplete"
        else:
            eval_result = "resolved" if resolved else "unresolved"
        _update_prediction(json_path, eval_result=eval_result)

        output = {}
        if output_path.exists():
            try:
                loaded = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                # SIGKILL during the evaluator's non-atomic JSON write can
                # leave a truncated _output.json. _load_partial_eval_results
                # already skips these; mirror that here so log writing for
                # other instances isn't aborted by one bad file.
                loaded = None
            if isinstance(loaded, dict):
                output = loaded
        tests_status = _build_tests_status(output, entries_by_id[instance_id])
        eval_text = _eval_log_text(resolved, tests_status)
        if eval_result in {"eval_timeout", "eval_error", "eval_incomplete"}:
            lines = []
            if eval_result == "eval_timeout":
                lines.append(
                    f"Official evaluation timed out after {eval_metadata.get('timeout_sec', 'unknown')} seconds."
                )
            elif eval_result == "eval_error":
                lines.append(
                    f"Official evaluation failed with return code {eval_metadata.get('returncode', 'unknown')}."
                )
            else:
                lines.append("Official evaluation did not produce an instance result.")
            stdout_tail = str(eval_metadata.get("stdout_tail") or "").strip()
            stderr_tail = str(eval_metadata.get("stderr_tail") or "").strip()
            if stdout_tail:
                lines.extend(["", "## Evaluator stdout tail", stdout_tail])
            if stderr_tail:
                lines.extend(["", "## Evaluator stderr tail", stderr_tail])
            eval_text = "\n".join(lines) + "\n\n" + eval_text
        (out_dname / f"{instance_filename}_eval.md").write_text(eval_text, encoding="utf-8")


_TOKEN_USAGE_RE = re.compile(r'^TOKEN_USAGE (\{.*\})\s*$', re.MULTILINE)


def _aggregate_token_usage(out_dname: Path) -> dict[str, Any]:
    """Scan per-task `.md` chat logs for TOKEN_USAGE records emitted by
    ``baselines/dgm/llm.log_token_usage`` and aggregate into a single
    ``llm_usage`` block for the harness report.

    Returns an empty dict if the directory has no per-task logs yet.
    """
    if out_dname is None or not out_dname.exists():
        return {}

    per_instance: dict[str, dict[str, int]] = {}
    totals = {
        "calls": 0,
        "input_tokens": 0,
        "cached_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }
    malformed = 0

    for md_path in sorted(out_dname.glob("instance_*.md")):
        if md_path.name.endswith("_eval.md"):
            continue
        instance_id = md_path.stem
        try:
            text = md_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        per = {k: 0 for k in totals}
        for match in _TOKEN_USAGE_RE.finditer(text):
            try:
                rec = json.loads(match.group(1))
            except json.JSONDecodeError:
                malformed += 1
                continue
            per["calls"] += 1
            for key in ("input_tokens", "cached_tokens", "output_tokens",
                        "reasoning_tokens", "total_tokens"):
                per[key] += int(rec.get(key) or 0)
        if per["calls"]:
            per_instance[instance_id] = per
            for k in totals:
                totals[k] += per[k]

    if not per_instance and not malformed:
        return {}
    return {
        **totals,
        "malformed_records": malformed,
        "per_instance": per_instance,
    }


def build_report(
    entries: list[dict[str, Any]], results: list[dict[str, Any]], eval_results: dict[str, bool],
    out_dname: Path | None = None,
    eval_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eval_metadata = eval_metadata or {}
    submitted_ids = [validate_instance_id(result["instance_id"]) for result in results]
    completed_ids = [validate_instance_id(result["instance_id"]) for result in results if result.get("success")]
    incomplete_ids = [validate_instance_id(result["instance_id"]) for result in results if not result.get("success")]
    # agent_timeout_ids counts instances where the agent was killed for time
    # AND emitted no usable patch. If the agent timed out mid-run but still
    # produced a non-empty patch, the instance still goes through official
    # eval and lands in resolved/unresolved/eval_*_ids — counting it here
    # too would double-count it against attempted_eval_ids.
    agent_timeout_ids = sorted(
        validate_instance_id(result["instance_id"])
        for result in results
        if result.get("success")
        and result.get("agent_status") == "timeout"
        and not str(result.get("eval_patch", result.get("model_patch")) or "").strip()
    )
    empty_patch_ids = [
        validate_instance_id(result["instance_id"])
        for result in results
        if result.get("success")
        and result.get("agent_status") != "timeout"
        and not str(result.get("eval_patch", result.get("model_patch")) or "").strip()
    ]
    eval_results = {validate_instance_id(instance_id): resolved for instance_id, resolved in eval_results.items()}
    resolved_ids = sorted(instance_id for instance_id, resolved in eval_results.items() if resolved)
    # unresolved means the official grader returned False, NOT "no result". An
    # instance whose eval timed out / errored / never ran shows up in
    # eval_incomplete_ids instead — counting it here too inflates the
    # unresolved tally past completed_instances.
    unresolved_ids = sorted(
        validate_instance_id(result["instance_id"])
        for result in results
        if result.get("success")
        and str(result.get("eval_patch", result.get("model_patch")) or "").strip()
        and validate_instance_id(result["instance_id"]) in eval_results
        and not eval_results[validate_instance_id(result["instance_id"])]
    )
    error_ids = sorted(set(incomplete_ids))
    attempted_eval_ids = sorted(
        validate_instance_id(result["instance_id"])
        for result in results
        if result.get("success") and str(result.get("eval_patch", result.get("model_patch")) or "").strip()
    )
    eval_incomplete_ids = sorted(instance_id for instance_id in attempted_eval_ids if instance_id not in eval_results)

    return {
        "total_instances": len(entries),
        "submitted_instances": len(submitted_ids),
        "completed_instances": len(completed_ids),
        "resolved_instances": len(resolved_ids),
        "unresolved_instances": len(unresolved_ids),
        "eval_incomplete_instances": len(eval_incomplete_ids),
        "agent_timeout_instances": len(agent_timeout_ids),
        "empty_patch_instances": len(empty_patch_ids),
        "error_instances": len(error_ids),
        "unstopped_instances": 0,
        "completed_ids": sorted(completed_ids),
        "incomplete_ids": sorted(incomplete_ids),
        "agent_timeout_ids": sorted(agent_timeout_ids),
        "empty_patch_ids": sorted(empty_patch_ids),
        "submitted_ids": sorted(submitted_ids),
        "resolved_ids": sorted(resolved_ids),
        "unresolved_ids": sorted(unresolved_ids),
        "eval_incomplete_ids": sorted(eval_incomplete_ids),
        "error_ids": sorted(error_ids),
        "unstopped_containers": [],
        "unremoved_images": [],
        "schema_version": "swebench_pro_v1",
        "llm_usage": _aggregate_token_usage(out_dname) if out_dname is not None else {},
        "official_eval_status": str(eval_metadata.get("status") or "unknown"),
        "official_eval_returncode": eval_metadata.get("returncode"),
        "official_eval_timeout_sec": eval_metadata.get("timeout_sec"),
        "official_eval_stdout_tail": str(eval_metadata.get("stdout_tail") or ""),
        "official_eval_stderr_tail": str(eval_metadata.get("stderr_tail") or ""),
        "official_eval_leaked_containers_stopped": int(eval_metadata.get("leaked_containers_stopped") or 0),
    }


def harness(
    dataset_path: str | Path | None = None,
    task_map: str | Path | None = None,
    test_task_list: list[str] | None = None,
    num_samples: int = -1,
    max_workers: int = 4,
    model_name_or_path: str | None = None,
    model_patch_paths: list[str] | None = None,
    num_evals: int = 1,
    num_evals_parallel: int = 1,
    pred_dname: str | Path = "./swe_bench_pro/predictions",
    output_dir: str | Path = "./swe_bench_pro/reports",
    eval_source: str | Path | None = None,
    scripts_dir: str | Path | None = None,
    dockerhub_username: str = DEFAULT_DOCKERHUB_USERNAME,
    use_local_docker: bool = True,
    docker_platform: str | None = None,
    block_network: bool = False,
) -> list[Path]:
    _load_shared_env()
    dataset_path = Path(dataset_path or DEFAULT_DATASET_PATH).resolve()
    task_map_path = Path(task_map).resolve() if task_map else None
    eval_source = Path(eval_source or DEFAULT_EVAL_SOURCE).resolve()
    scripts_dir = Path(scripts_dir or eval_source / "run_scripts").resolve()
    pred_dname = Path(pred_dname)
    output_dir = Path(output_dir)
    pred_dname.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = _read_jsonl(dataset_path)
    task_ids = test_task_list if test_task_list is not None else load_task_ids(task_map_path)
    entries = _select_entries(entries, task_ids)
    if num_samples > 0:
        entries = entries[:num_samples]
    if not entries:
        raise ValueError("No SWE-bench Pro entries selected")

    if model_name_or_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_or_path = f"{timestamp}--dgm"

    out_dnames = []
    entries_by_id = {validate_instance_id(entry["instance_id"]): entry for entry in entries}
    for eval_idx in range(num_evals):
        model_name_or_path_inst = f"{model_name_or_path}_{eval_idx}"
        out_dname = pred_dname / model_name_or_path_inst
        out_dname.mkdir(parents=True, exist_ok=True)
        out_dnames.append(out_dname)

        results = []
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            future_to_entry = {
                executor.submit(
                    process_entry,
                    entry,
                    out_dname,
                    model_name_or_path_inst,
                    model_patch_paths,
                    scripts_dir=scripts_dir,
                    dockerhub_username=dockerhub_username,
                    docker_platform=docker_platform,
                ): entry
                for entry in entries
            }
            for future in as_completed(future_to_entry):
                result = future.result()
                results.append(result)
                if result.get("success"):
                    print(f"Processed {result['instance_id']} for eval {eval_idx}")
                else:
                    print(f"Failed {result['instance_id']} for eval {eval_idx}: {result.get('error', 'unknown')}")

        patch_bundle = out_dname / "swebench_pro_patches.json"
        patch_payload = _write_patch_bundle(patch_bundle, results, model_name_or_path_inst, entries_by_id)
        official_eval_dir = output_dir / "official_eval" / model_name_or_path_inst
        eval_results: dict[str, bool] = {}
        eval_metadata: dict[str, Any] = {"status": "skipped", "timeout_sec": DEFAULT_OFFICIAL_EVAL_TIMEOUT_SEC}
        if patch_payload:
            eval_metadata = _run_official_eval(
                patch_bundle=patch_bundle,
                official_eval_dir=official_eval_dir,
                dataset_path=dataset_path,
                eval_source=eval_source,
                scripts_dir=scripts_dir,
                dockerhub_username=dockerhub_username,
                max_workers=min(max_workers, len(patch_payload)),
                use_local_docker=use_local_docker,
                docker_platform=docker_platform,
                block_network=block_network,
            )
            eval_results = _load_eval_results(official_eval_dir)
            partial_eval_results = _load_partial_eval_results(
                official_eval_dir,
                prefix=model_name_or_path_inst,
                entries_by_id=entries_by_id,
            )
            for instance_id, resolved in partial_eval_results.items():
                eval_results.setdefault(instance_id, resolved)

        _write_eval_logs(
            entries_by_id=entries_by_id,
            results=results,
            out_dname=out_dname,
            official_eval_dir=official_eval_dir,
            prefix=model_name_or_path_inst,
            eval_results=eval_results,
            eval_metadata=eval_metadata,
        )
        report = build_report(entries, results, eval_results, out_dname=out_dname, eval_metadata=eval_metadata)
        report_file = output_dir / f"{model_name_or_path.replace('/', '__')}_{eval_idx}.000.json"
        report_file.write_text(json.dumps(report, indent=4), encoding="utf-8")
        print(f"Report written to {report_file}")

    return out_dnames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--task-map", type=Path, default=DEFAULT_TASK_MAP)
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--model-name", "--model-name-or-path", dest="model_name_or_path", default=None)
    parser.add_argument("--model-patch-paths", default=None, help="Comma-separated DGM model patches.")
    parser.add_argument("--num-evals", type=int, default=1)
    parser.add_argument("--pred-dname", type=Path, default=Path("./swe_bench_pro/predictions"))
    parser.add_argument("--output-dir", type=Path, default=Path("./swe_bench_pro/reports"))
    parser.add_argument("--eval-source", type=Path, default=DEFAULT_EVAL_SOURCE)
    parser.add_argument("--scripts-dir", type=Path, default=None)
    parser.add_argument("--dockerhub-username", default=DEFAULT_DOCKERHUB_USERNAME)
    parser.add_argument(
        "--no-local-docker", action="store_true", help="Use Modal instead of local Docker for official eval."
    )
    parser.add_argument("--docker-platform", default=None)
    parser.add_argument(
        "--block-network", action="store_true", help="Block network during official evaluation containers."
    )
    args = parser.parse_args()

    model_patch_paths = args.model_patch_paths.split(",") if args.model_patch_paths else None
    harness(
        dataset_path=args.dataset_path,
        task_map=args.task_map,
        num_samples=args.num_samples,
        max_workers=args.max_workers,
        model_name_or_path=args.model_name_or_path,
        model_patch_paths=model_patch_paths,
        num_evals=args.num_evals,
        pred_dname=args.pred_dname,
        output_dir=args.output_dir,
        eval_source=args.eval_source,
        scripts_dir=args.scripts_dir,
        dockerhub_username=args.dockerhub_username,
        use_local_docker=not args.no_local_docker,
        docker_platform=args.docker_platform,
        block_network=args.block_network,
    )


if __name__ == "__main__":
    main()
