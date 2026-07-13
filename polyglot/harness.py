import argparse
import datetime
import glob
import json
import os
import tempfile
from enum import Enum
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import docker
from datasets import load_dataset
from dotenv import load_dotenv

from prompts.testrepo_prompt import get_test_description
from polyglot.test_spec import make_test_spec
from polyglot.docker_build import build_env_images, build_container, cleanup_container
from polyglot.constants import MAP_REPO_VERSION_TO_SPECS, TEST_COMMANDS
from polyglot import leak_scrub
from utils.git_utils import filter_patch_by_files, remove_patch_by_files

from swe_bench.utils import (
    copy_to_container,
    copy_from_container,
    log_container_output,
    remove_existing_container,
    safe_log,
    setup_logger,
)


def _polyglot_agent_timeout_sec(default: int = 600) -> int:
    """Resolve the polyglot agent timeout (seconds).

    Checks CROSS_RUNNER_AGENT_TIMEOUT_SEC first (sweep-level override shared
    across all runners), then DGM_POLYGLOT_AGENT_TIMEOUT_SEC (DGM-specific),
    falling back to ``default`` (the legacy 600s / 10-min budget) when neither
    is set to a positive integer. Mirrors swe_bench.pro_harness'
    _cross_runner_agent_timeout_sec so the polyglot harness honors the same
    sweep-level timeout as the SWE-bench Pro harness (kcsi #1125).
    """
    for var in ("CROSS_RUNNER_AGENT_TIMEOUT_SEC", "DGM_POLYGLOT_AGENT_TIMEOUT_SEC"):
        raw = os.environ.get(var, "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value > 0:
            return value
    return default


def _load_shared_env() -> None:
    """Load swarms-side shared env files but NEVER clobber wrapper-set DGM_*.

    Wrapper scripts export DGM_CLAUDE_MODEL / DGM_OPENAI_MODEL / DGM_CODE_MODEL /
    DGM_SELF_IMPROVE_MODEL / DGM_DIAGNOSE_MODEL / DGM_REASONING_EFFORT before
    invoking DGM. Without this snapshot, the subsequent load_dotenv(..., override=True)
    would re-read the static defaults in shared.env / .env.openai (e.g.
    DGM_OPENAI_MODEL=gpt-5.4-mini) and silently overwrite the wrapper's per-sweep
    choice. See llm.py for the original PR #501 fix that this mirrors.
    """
    repo_root = Path(__file__).resolve().parents[3]
    env_paths = [
        repo_root / "configs" / "providers" / ".env.shared",
        repo_root / "configs" / "providers" / ".env.haiku",
        repo_root / "configs" / "providers" / ".env.openai",
        repo_root / "configs" / "models" / "shared.env",
    ]
    authoritative_keys = (
        "DGM_CLAUDE_MODEL",
        "DGM_OPENAI_MODEL",
        "DGM_CODE_MODEL",
        "DGM_SELF_IMPROVE_MODEL",
        "DGM_DIAGNOSE_MODEL",
        "DGM_REASONING_EFFORT",
    )
    snapshot = {k: os.environ.get(k) for k in authoritative_keys if os.environ.get(k) is not None}
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=True)
    for k, v in snapshot.items():
        os.environ[k] = v


def _collect_runtime_env(names):
    env_vars = {}
    for name in names:
        value = os.getenv(name)
        if value:
            env_vars[name] = value
    return env_vars

def get_eval_script(commands):
    return "\n".join(["#!/bin/bash", "set -uxo pipefail"] + commands) + "\n"

# --- Vacuous-pass guard (parity with the KCSI polyglot grader, kcsi #1137) ---
# A language test runner can exit 0 without any test actually executing (e.g. a
# solution neutralised the official test/build config). The KCSI polyglot
# harness (src/kcsi/eval/polyglot_harness.py) flips such a "pass" to a fail for
# the languages that emit a reliable zero-tests-ran signal. We mirror that here
# so this baseline's polyglot scores are graded on the same footing as KCSI.
#
# Only go/rust/python have robust markers under this harness's TEST_COMMANDS.
# cpp runs a build-time Catch2 via cmake+make (no ctest marker to key on) and,
# together with js/java, is left to the existing `git reset --hard <test_commit>`
# restore of the official test/build files -- matching KCSI, which likewise
# leaves js/java unguarded.
_GO_NO_TEST_FILES_MARKER = "[no test files]"
_GO_TESTS_RAN_RE = re.compile(r"(?m)^ok\s")
_RUST_TESTS_RAN_RE = re.compile(r"test result: ok\.\s+[1-9]\d*\s+passed")
_PYTEST_NO_TESTS_MARKERS = ("no tests ran", "collected 0 items")


def is_vacuous_pass(language, eval_output):
    """Return True if an exit-0 polyglot eval ran zero tests (a vacuous pass).

    Mirrors the KCSI polyglot vacuous-pass guard (kcsi #1137) for the languages
    with a reliable zero-tests-ran signal. Only ever turns a *pass* into a fail,
    never the reverse.
    """
    text = eval_output or ""
    if language == "go":
        # "[no test files]" with no "ok\t<pkg>" line -> no package ran a test.
        return _GO_NO_TEST_FILES_MARKER in text and not _GO_TESTS_RAN_RE.search(text)
    if language == "rust":
        # A genuine pass reports "test result: ok. <n>=1.. passed"; absence on an
        # exit-0 run means "test result: ok. 0 passed" (zero tests ran).
        return not _RUST_TESTS_RAN_RE.search(text)
    if language == "python":
        # pytest normally exits 5 on "no tests collected"; these markers only
        # catch a run forced to exit 0 while collecting nothing.
        return any(marker in text for marker in _PYTEST_NO_TESTS_MARKERS)
    return False


def process_entry(entry, out_dname, model_name_or_path, model_patch_paths):
    """
    Process a single dataset entry. This function encapsulates the main processing logic
    for each entry to make it suitable for parallel execution.
    """
    instance_id = entry['instance_id']
    problem_statement = entry['problem_statement']
    base_commit = entry['base_commit']
    chat_history_file = out_dname / (instance_id + ".md")
    out_fname = out_dname / (instance_id + ".json")
    eval_file = out_dname / f"{instance_id}_eval.sh"
    eval_result_file = out_dname / f"{instance_id}_eval.md"

    # Skip if output result already exists
    if out_fname.exists():
        print(f"Skipping existing entry {instance_id}")
        with open(out_fname) as f:
            result = json.loads(f.read())
        return result

    try:
        _load_shared_env()
        # Create and start the Docker container
        client = docker.from_env()
        run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        # Set up thread-specific logger
        logger = setup_logger(str(out_dname / f"{instance_id}_docker.log"))
        nocache = True
        test_spec = make_test_spec(entry)
        # Remove any existing container with the same name
        container_name = test_spec.get_instance_container_name(run_id)
        remove_existing_container(client, container_name)
        # Now create and start the container
        container = build_container(test_spec, client, run_id, logger, nocache, force_rebuild=False)
        container.start()

        # --- History-leak lockdown (mirror kcsi sanitizeRepoHistory, issue #924) ---
        # /testbed is a local `git clone` of the per-exercise repo, so its object
        # store + origin refs still reach test_commit (the hidden tests + the
        # .meta/example reference solution); the /polyglot clone source keeps them
        # on disk too. Scrub /testbed down to base_commit and remove /polyglot
        # BEFORE the agent's unrestricted bash tool runs. Runs as root (removing
        # the root-owned /polyglot and gc-ing need it; it re-chmods 777 after).
        # Fail closed: if the scrub cannot be verified, abort the task rather than
        # run the agent against a leaky workspace.
        container_repo = "polyglot" if Path(entry["repo"]).is_absolute() else entry["repo"]
        clone_source = f"/{container_repo}"
        lockdown_script = leak_scrub.pre_agent_lockdown_script(
            "/testbed", base_commit, entry["test_commit"], clone_source
        )
        exec_result = container.exec_run(
            ["/bin/bash", "-c", lockdown_script], workdir="/", user="root"
        )
        log_container_output(exec_result)
        if exec_result.exit_code != 0:
            raise RuntimeError(
                f"History-leak lockdown failed for {instance_id}; refusing to run "
                f"the agent against a leaky workspace: "
                f"{exec_result.output.decode(errors='replace')[-500:]}"
            )

        # Copy the necessary files and requirements to the container
        copy_to_container(container, 'coding_agent_polyglot.py', '/dgm/coding_agent.py')
        copy_to_container(container, 'requirements.txt', '/dgm/requirements.txt')
        copy_to_container(container, 'pytest.ini', '/dgm/pytest.ini')
        copy_to_container(container, 'tools/', '/dgm/tools/')
        copy_to_container(container, 'utils/', '/dgm/utils/')
        copy_to_container(container, 'tests/', '/dgm/tests/')
        copy_to_container(container, 'prompts/', '/dgm/prompts/')
        copy_to_container(container, 'llm.py', '/dgm/llm.py')
        copy_to_container(container, 'llm_withtools.py', '/dgm/llm_withtools.py')
        chat_history_file_container = f'/dgm/{chat_history_file.name}'

        # See the checked repo
        exec_result = container.exec_run("ls -R /testbed", workdir='/')
        log_container_output(exec_result)

        # Get test description
        eval_cmd = MAP_REPO_VERSION_TO_SPECS[entry['language']]['test_cmd']
        test_description = get_test_description(eval_cmd, polyglot=True)

        # Apply model patch
        if model_patch_paths:
            safe_log("Applying model patches")
            for model_patch_path in model_patch_paths:
                copy_to_container(container, model_patch_path, '/dgm/parent_patch.txt')
                exec_result = container.exec_run("/bin/sh -c 'patch -p1 < /dgm/parent_patch.txt'", workdir='/dgm')
                log_container_output(exec_result)
                exec_result = container.exec_run("rm /dgm/parent_patch.txt", workdir='/dgm')
                log_container_output(exec_result)

        # Install this repo requirements
        safe_log("Installing more requirements")
        exec_result = container.exec_run("python -m pip install -r /dgm/requirements.txt", workdir='/')
        log_container_output(exec_result)

        # Run the agent
        env_vars = _collect_runtime_env([
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "OPENROUTER_API_KEY",
            "DEEPSEEK_API_KEY",
            "AWS_REGION",
            "AWS_REGION_NAME",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "DGM_CLAUDE_MODEL",
            "DGM_OPENAI_MODEL",
            "DGM_CODE_MODEL",
            "DGM_SELF_IMPROVE_MODEL",
            "DGM_DIAGNOSE_MODEL",
            "DGM_REASONING_EFFORT",
            "OPENAI_REASONING_EFFORT",
            "REASONING_EFFORT",
        ])
        safe_log("Running the agent")
        cmd = [
            "timeout", str(_polyglot_agent_timeout_sec()),  # default 10 min; CROSS_RUNNER_AGENT_TIMEOUT_SEC overrides
            "python", "/dgm/coding_agent.py",
            "--problem_statement", problem_statement,
            "--git_dir", "/testbed/",
            "--chat_history_file", chat_history_file_container,
            "--base_commit", base_commit,
            "--outdir", "/dgm/",
            "--test_description", test_description,
            "--language", entry['language'],
        ]
        exec_result = container.exec_run(cmd, environment=env_vars, workdir='/testbed/')
        log_container_output(exec_result)

        # Copy output files back to host
        logger.info("Copying output files back to host")
        copy_from_container(container, chat_history_file_container, chat_history_file)
        # Additional chat history files
        exec_result = container.exec_run(f"find /dgm/ -name '{instance_id}_*.md'", workdir='/')
        chat_history_files_container = exec_result.output.decode().split()
        for chat_history_file_container in chat_history_files_container:
            chat_history_file = out_dname / Path(chat_history_file_container).name
            copy_from_container(container, chat_history_file_container, chat_history_file)

        # Get model_patch
        logger.info("Getting model_patch")
        exec_result = container.exec_run("cat /dgm/model_patch.diff")
        log_container_output(exec_result)
        model_patch = ''
        model_patch = exec_result.output.decode()

        # Additional proposed model patches
        proposed_model_patches = []

        # Directly do eval
        eval_result = ''
        if not model_patch:
            eval_result = 'empty_patch'
            result = {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": model_patch,
            'proposed_model_patches': proposed_model_patches,
            "eval_result": eval_result,
            "success": True
        }
            out_fname.write_text(json.dumps(result, indent=4))
            return {"success": True, "instance_id": instance_id, "eval_result": eval_result}


        # Re-inject the hidden-test history that the pre-agent lockdown scrubbed
        # from /testbed, from a git bundle built on the HOST (a source the agent
        # could not read during its run). This restores test_commit so the
        # grader's `git reset --hard {test_commit}` below works byte-for-byte as
        # before. Fail closed: a failed re-injection would silently grade against
        # base_commit (no hidden tests), inflating scores.
        grade_bundle = out_dname / f"{instance_id}_grade_testcommit.bundle"
        leak_scrub.build_grade_bundle(entry["repo"], grade_bundle, entry["test_commit"])
        copy_to_container(container, str(grade_bundle), "/tmp/grade_testcommit.bundle")
        inject_script = leak_scrub.inject_grade_bundle_script(
            "/testbed", "/tmp/grade_testcommit.bundle", entry["test_commit"]
        )
        exec_result = container.exec_run(["/bin/bash", "-c", inject_script], workdir="/")
        log_container_output(exec_result)
        if exec_result.exit_code != 0:
            raise RuntimeError(
                f"Grade-time hidden-test re-injection failed for {instance_id}: "
                f"{exec_result.output.decode(errors='replace')[-500:]}"
            )
        container.exec_run("rm -f /tmp/grade_testcommit.bundle", workdir="/")
        try:
            grade_bundle.unlink()
        except OSError:
            pass

        exec_result = container.exec_run("git -C /testbed stash push " + " ".join(entry['files']['solution']), workdir='/')
        log_container_output(exec_result)
        exec_result = container.exec_run(f"git -C /testbed reset --hard {entry['test_commit']}", workdir='/')
        log_container_output(exec_result)
        exec_result = container.exec_run(f"git -C /testbed clean -fd", workdir='/')
        log_container_output(exec_result)
        exec_result = container.exec_run("git -C /testbed stash pop", workdir='/')
        log_container_output(exec_result)

        safe_log("Running the eval")
        language = entry['language']
        test_command = TEST_COMMANDS[language]

        eval_file.write_text(get_eval_script(test_command))

        copy_to_container(container, eval_file, '/testbed/eval.sh')
        exec_result = container.exec_run("ls -R /testbed", workdir='/')
        log_container_output(exec_result)
        exec_result = container.exec_run("chmod +x /testbed/eval.sh", workdir='/')
        log_container_output(exec_result)

        # Unify the polyglot eval (test-execution) timeout at 180s to match the
        # KCSI polyglot harness (--polyglot-timeout-sec 180); compiled-language
        # test suites need more than the legacy 120s budget (kcsi #1196).
        exec_result = container.exec_run("timeout 180 ./eval.sh", workdir='/testbed')
        log_container_output(exec_result, raise_error=False)
        eval_output = exec_result.output.decode(errors='replace')
        eval_result_file.write_text(eval_output)
        if exec_result.exit_code != 0:
            eval_result = 'unresolved'
        elif is_vacuous_pass(language, eval_output):
            safe_log(
                f"Vacuous-pass guard: {language} eval for {instance_id} exited 0 "
                f"but ran zero tests; scoring 'unresolved' (kcsi #1137 parity)"
            )
            eval_result = 'unresolved'
        else:
            eval_result = 'resolved'

        # Write result to file
        result = {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": model_patch,
            'proposed_model_patches': proposed_model_patches,
            "eval_result": eval_result,
            "success": True
        }
        out_fname.write_text(json.dumps(result, indent=4))

        return {"success": True, "instance_id": instance_id, "eval_result": eval_result}

    except Exception as e:

        # Check if eval_result exists in local scope
        if 'eval_result' not in locals():
            eval_result = 'incomplete'
        else:
            eval_result = 'error'
        if 'model_patch' not in locals():
            model_patch = ''
        if 'proposed_model_patches' not in locals():
            proposed_model_patches = ''

        # Write result to file
        result = {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": model_patch,
            'proposed_model_patches': proposed_model_patches,
            "eval_result": eval_result,
            "success": False
        }
        out_fname.write_text(json.dumps(result, indent=4))

        print(f"Error processing entry {instance_id}: {str(e)}")
        return {"success": False, "instance_id": instance_id, "eval_result": eval_result}

    finally:
        # Clean up docker container
        try:
            cleanup_container(client, container, logger)
        except Exception as e:
            print(f"Error cleaning up Docker container for {instance_id}: {e}")


def _aggregate_token_usage(pred_dir):
    """Aggregate TOKEN_USAGE records from per-task markdown chat histories.

    DGM's ``llm.py`` / ``llm_withtools.py`` emit ``TOKEN_USAGE {...json...}``
    lines into each task's chat markdown under ``pred_dir``. This mirrors the
    swebench_pro TOKEN_USAGE aggregator (DGM PR #11) so polyglot report.json
    exposes the same ``llm_usage`` block that downstream sweep aggregation
    expects from every benchmark.
    """
    aggregate = {
        "calls": 0,
        "malformed_records": 0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "cached_prompt_tokens": 0,
        "uncached_prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
    }
    saw_cost = False
    if pred_dir is None:
        aggregate["cost_usd"] = None
        return aggregate

    md_files = sorted(glob.glob(os.path.join(str(pred_dir), "*.md")))
    for md_path in md_files:
        try:
            with open(md_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if "TOKEN_USAGE" not in line:
                        continue
                    split_result = line.split("TOKEN_USAGE", 1)
                    payload = split_result[1].lstrip(": ").strip()
                    try:
                        record = json.loads(payload)
                    except json.JSONDecodeError:
                        aggregate["malformed_records"] += 1
                        continue

                    input_tokens = int(record.get("input_tokens") or 0)
                    output_tokens = int(record.get("output_tokens") or 0)
                    total_tokens = int(record.get("total_tokens") or 0)
                    cached_tokens = int(record.get("cached_tokens") or 0)
                    reasoning_tokens = int(record.get("reasoning_tokens") or 0)

                    aggregate["calls"] += 1
                    aggregate["prompt_tokens"] += input_tokens
                    aggregate["cached_prompt_tokens"] += cached_tokens
                    aggregate["uncached_prompt_tokens"] += max(input_tokens - cached_tokens, 0)
                    aggregate["completion_tokens"] += output_tokens
                    aggregate["total_tokens"] += total_tokens or input_tokens + output_tokens
                    aggregate["reasoning_tokens"] += reasoning_tokens

                    cost = record.get("cost_usd")
                    if isinstance(cost, (int, float)):
                        aggregate["cost_usd"] += float(cost)
                        saw_cost = True
        except OSError:
            continue

    if not saw_cost:
        aggregate["cost_usd"] = None
    return aggregate


def build_report(entries, results, pred_dir=None):
    incomplete_ids = [result["instance_id"] for result in results if not result["success"]]
    completed_ids = [result["instance_id"] for result in results if result["success"]]
    resolved_ids = []
    unresolved_ids = []
    error_ids = []
    empty_patch_ids = []
    unstopped_containers = []
    unremoved_images = []

    for result in results:
        if result["success"]:
            if result.get("eval_result") == "resolved":
                resolved_ids.append(result["instance_id"])
            elif result.get("eval_result") == "unresolved":
                unresolved_ids.append(result["instance_id"])
            elif result.get("eval_result") == "empty_patch":
                empty_patch_ids.append(result["instance_id"])
            else:
                error_ids.append(result["instance_id"])

    return {
        "total_instances": len(entries),
        "submitted_instances": len(results),
        "completed_instances": len(completed_ids),
        "resolved_instances": len(resolved_ids),
        "unresolved_instances": len(unresolved_ids),
        "empty_patch_instances": len(empty_patch_ids),
        "error_instances": len(error_ids),
        "unstopped_instances": len(unstopped_containers),
        "completed_ids": list(sorted(completed_ids)),
        "incomplete_ids": list(sorted(incomplete_ids)),
        "empty_patch_ids": list(sorted(empty_patch_ids)),
        "submitted_ids": list(sorted(result["instance_id"] for result in results)),
        "resolved_ids": list(sorted(resolved_ids)),
        "unresolved_ids": list(sorted(unresolved_ids)),
        "error_ids": list(sorted(error_ids)),
        "unstopped_containers": list(sorted(unstopped_containers)),
        "unremoved_images": list(sorted(unremoved_images)),
        "llm_usage": _aggregate_token_usage(pred_dir),
        "schema_version": 2,
    }

def harness(
        dataset_path=None,
        test_task_list=None,
        num_samples=-1,
        max_workers=4,
        model_name_or_path=None,
        model_patch_paths=None,
        num_evals=1,
        num_evals_parallel=1,
        pred_dname='./polyglot/predictions',
        output_dir='./polyglot/predictions'
    ):
    """
    _load_shared_env()
    Parallel processing harness using ThreadPoolExecutor.

    Args:
        test_task_list: List of task IDs to process (None for all)
        num_samples: Number of samples to process (-1 for all)
        max_workers: Maximum number of concurrent threads
        model_name_or_path: Model name or path
        model_patch_paths: Paths to the model patches for dgm
        num_evals: Repeated number of swe evaluations
    """
    _load_shared_env()
    if dataset_path is None:
        dataset_path = os.getenv(
            "DGM_POLYGLOT_METADATA",
            "../../benchmarks/polyglot/source/polyglot_benchmark_metadata.json",
        )

    if model_patch_paths:
        for model_patch_path in model_patch_paths:
            # Read and modify model patch
            with open(model_patch_path, 'r') as f:
                patch_content = f.read()
            patch_content = remove_patch_by_files(patch_content)
            # Placeholder for any patch modifications if needed
            with open(model_patch_path, 'w') as f:
                f.write(patch_content)

    if num_evals > 1:
        raise ValueError("Multiple evaluations (num_evals > 1) is not supported with polyglot")

    # Load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Ensure that necessary directories exist
    if model_name_or_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_or_path = f"{timestamp}--claude-3-5-sonnet-20241022"
    pred_dname = Path(pred_dname)
    pred_dname.mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_dnames = []

    # Prepare the dataset entries
    entries = list(dataset)
    if test_task_list:
        entries = [entry for entry in entries if entry['instance_id'] in test_task_list]
    if num_samples > 0:
        entries = entries[:num_samples]

    # Build the environment images
    client = docker.from_env()
    build_env_images(client, dataset=entries, max_workers=max_workers, force_rebuild=False)

    # Define a function to handle a single evaluation for all specified issues
    def process_evaluation(eval_idx):
        model_name_or_path_inst = f"{model_name_or_path}_{eval_idx}"
        out_dname = pred_dname / model_name_or_path_inst
        out_dname.mkdir(exist_ok=True)

        print(f"Starting evaluation {eval_idx} for model {model_name_or_path}")

        # Process entries in parallel
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_entry = {
                executor.submit(process_entry, entry, out_dname, model_name_or_path_inst, model_patch_paths): entry
                for entry in entries
            }

            # Process completed tasks as they finish
            for future in as_completed(future_to_entry):
                result = future.result()
                results.append(result)
                if result["success"]:
                    print(f"Successfully processed entry {result['instance_id']} for eval {eval_idx}")
                else:
                    print(f"Failed to process entry {result['instance_id']} for eval {eval_idx}: {result.get('error', 'Unknown error')}")
        # Get final results from completed futures

        return out_dname, results

    out_dname, results = process_evaluation(0)
    print(f"All evaluations completed for model {model_name_or_path}")

    # Directly generate report
    report = build_report(entries, results, pred_dir=out_dname)

    print(report)
    report_file = output_dir / Path(
        model_name_or_path.replace("/", "__") + f"_{0}"
        + f".{000}"
        + ".json"
    )
    with open(report_file, "w") as f:
        print(json.dumps(report, indent=4), file=f)
    print(f"Report written to {report_file}")

    return out_dnames

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to process")
    parser.add_argument("--max_workers", type=int, default=5, help="Maximum number of concurrent threads")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Model name or path")
    parser.add_argument("--model_patch_paths", type=str, default=None, help="Paths to the model patches")
    parser.add_argument("--num_evals", type=int, default=1, help="Repeated number of swe evaluations")
    parser.add_argument("--num_evals_parallel", type=int, default=1, help="Number of parallel repeated evaluations")
    args = parser.parse_args()

    dataset_path = os.getenv(
        "DGM_POLYGLOT_METADATA",
        "../../benchmarks/polyglot/source/polyglot_benchmark_metadata.json",
    )
    with open(dataset_path) as f:
        metadata = json.loads(f.read())
        language_task_list = [entry["instance_id"] for entry in metadata if entry["instance_id"].startswith("python")]
        # Create a list of all tasks from metadata
        all_task_list = [entry["instance_id"] for entry in metadata]

    from utils.common_utils import load_json_file
    swe_issues_med = load_json_file("./polyglot/subsets/medium.json")
    model_patch_paths = args.model_patch_paths.split(',') if args.model_patch_paths is not None else None
    # Run the parallel harness

    harness(
        dataset_path=dataset_path,
        test_task_list=all_task_list,
        num_samples=args.num_samples,
        max_workers=args.max_workers,
        model_name_or_path=args.model_name_or_path,
        model_patch_paths=model_patch_paths,
        num_evals=args.num_evals,
        num_evals_parallel=args.num_evals_parallel,
    )

if __name__ == "__main__":
    main()
