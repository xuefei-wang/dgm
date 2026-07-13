"""Pure-git (no Docker) tests for the polyglot workspace history-leak fix.

Reproduces the ``prepare_polyglot_dataset.register_git`` ->
``test_spec.make_repo_script_list`` (clone + reset) sequence in a tmp repo and
asserts that ``polyglot.leak_scrub`` closes every offline recovery vector for
the hidden tests + ``.meta/example`` reference solution, while keeping
``base_commit`` intact and grade-time test re-injection working.
"""

import importlib.util
import subprocess
from pathlib import Path

import pytest


DGM_ROOT = Path(__file__).resolve().parents[1]


def _load_leak_scrub():
    spec = importlib.util.spec_from_file_location(
        "polyglot_leak_scrub", DGM_ROOT / "polyglot" / "leak_scrub.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


leak_scrub = _load_leak_scrub()


HIDDEN_TEST_FILE = "test_solution.py"
HIDDEN_TEST_SECRET = "HIDDEN_TEST_SECRET_ASSERTION"
META_EXAMPLE_FILE = ".meta/example.py"
META_SECRET = "META_REFERENCE_SOLUTION_SECRET"
SOLUTION_FILE = "solution.py"


def _git(cwd, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _run_bash(script):
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True
    )


def _prepare_exercise(exercise_dir):
    """Mirror register_git: commit solution stubs (base_commit), then amend in
    the hidden tests + .meta example (test_commit)."""
    exercise_dir.mkdir(parents=True)
    _git(exercise_dir, "init", "-q")
    _git(exercise_dir, "config", "user.name", "prep")
    _git(exercise_dir, "config", "user.email", "prep@example.com")

    (exercise_dir / SOLUTION_FILE).write_text("def solve():\n    pass  # stub\n")
    (exercise_dir / ".docs").mkdir()
    (exercise_dir / ".docs" / "instructions.md").write_text("do the exercise\n")
    _git(exercise_dir, "add", SOLUTION_FILE, ".docs")
    _git(exercise_dir, "commit", "-qm", "Initial commit")
    base_commit = _git(exercise_dir, "rev-parse", "HEAD").stdout.strip()

    (exercise_dir / HIDDEN_TEST_FILE).write_text(
        f"assert solve() == 42  # {HIDDEN_TEST_SECRET}\n"
    )
    (exercise_dir / ".meta").mkdir()
    (exercise_dir / META_EXAMPLE_FILE).write_text(
        f"def solve():\n    return 42  # {META_SECRET}\n"
    )
    _git(exercise_dir, "add", ".")
    _git(exercise_dir, "commit", "-q", "--amend", "-m", "all files")
    test_commit = _git(exercise_dir, "rev-parse", "HEAD").stdout.strip()
    return base_commit, test_commit


def _clone_testbed(exercise_dir, testbed_dir, base_commit):
    """Mirror make_repo_script_list: local clone then reset --hard base_commit."""
    _git(testbed_dir.parent, "clone", "-q", str(exercise_dir), str(testbed_dir))
    _git(testbed_dir, "reset", "--hard", base_commit)


@pytest.fixture
def workspace(tmp_path):
    exercise_dir = tmp_path / "polyglot"  # the /polyglot clone source
    base_commit, test_commit = _prepare_exercise(exercise_dir)
    testbed_dir = tmp_path / "testbed"
    _clone_testbed(exercise_dir, testbed_dir, base_commit)
    return {
        "tmp_path": tmp_path,
        "exercise_dir": exercise_dir,
        "testbed_dir": testbed_dir,
        "base_commit": base_commit,
        "test_commit": test_commit,
    }


def test_fixture_actually_leaks_before_scrub(workspace):
    """Guard: the reproduction must genuinely leak, else the scrub asserts pass
    vacuously."""
    testbed = workspace["testbed_dir"]
    test_commit = workspace["test_commit"]

    # origin still reaches test_commit's hidden test.
    shown = _git(testbed, "show", f"origin/HEAD:{HIDDEN_TEST_FILE}", check=False)
    assert HIDDEN_TEST_SECRET in shown.stdout
    # test_commit object present -> cat-file recovers the .meta reference too.
    meta = _git(testbed, "cat-file", "-p", f"{test_commit}:{META_EXAMPLE_FILE}", check=False)
    assert META_SECRET in meta.stdout
    # clone source has the tests on disk.
    assert HIDDEN_TEST_SECRET in (workspace["exercise_dir"] / HIDDEN_TEST_FILE).read_text()


def test_lockdown_closes_every_recovery_vector(workspace):
    testbed = workspace["testbed_dir"]
    base_commit = workspace["base_commit"]
    test_commit = workspace["test_commit"]
    src = workspace["exercise_dir"]

    script = leak_scrub.pre_agent_lockdown_script(
        str(testbed), base_commit, test_commit, str(src)
    )
    result = _run_bash(script)
    assert result.returncode == 0, f"lockdown failed: {result.stdout}\n{result.stderr}"
    assert "LOCKDOWN-OK" in result.stdout

    # Every offline recovery vector for the hidden material is closed.
    assert _git(testbed, "show", f"origin/HEAD:{HIDDEN_TEST_FILE}", check=False).returncode != 0
    assert _git(testbed, "cat-file", "-p", f"{test_commit}:{HIDDEN_TEST_FILE}", check=False).returncode != 0
    assert _git(testbed, "cat-file", "-e", f"{test_commit}^{{commit}}", check=False).returncode != 0
    assert _git(testbed, "rev-list", "--all", "--count").stdout.strip() == "1"
    assert _git(testbed, "rev-list", "--all", "--not", "HEAD", "--count").stdout.strip() == "0"
    assert _git(testbed, "log", "--all", "--oneline").stdout.strip().count("\n") == 0
    assert _git(testbed, "reflog", check=False).stdout.strip() == ""
    assert _git(testbed, "remote").stdout.strip() == ""
    unreachable = [
        line
        for line in _git(testbed, "fsck", "--unreachable", "--no-dangling", check=False).stdout.splitlines()
        if line.startswith("unreachable ")
    ]
    assert unreachable == []
    # Clone source is gone (its on-disk tests were a second surface).
    assert not src.exists()


def test_lockdown_keeps_base_commit_intact(workspace):
    testbed = workspace["testbed_dir"]
    base_commit = workspace["base_commit"]
    test_commit = workspace["test_commit"]
    src = workspace["exercise_dir"]

    _run_bash(
        leak_scrub.pre_agent_lockdown_script(str(testbed), base_commit, test_commit, str(src))
    )

    assert _git(testbed, "rev-parse", "HEAD").stdout.strip() == base_commit
    # base_commit's solution stub is present; the hidden test is not tracked.
    assert "stub" in _git(testbed, "show", f"HEAD:{SOLUTION_FILE}").stdout
    assert _git(testbed, "cat-file", "-e", f"HEAD:{HIDDEN_TEST_FILE}", check=False).returncode != 0
    # Working tree matches base_commit: solution stub present, hidden test absent.
    assert (testbed / SOLUTION_FILE).exists()
    assert not (testbed / HIDDEN_TEST_FILE).exists()


def test_grade_reinjection_restores_hidden_tests_over_agent_solution(workspace, tmp_path):
    testbed = workspace["testbed_dir"]
    base_commit = workspace["base_commit"]
    test_commit = workspace["test_commit"]
    src = workspace["exercise_dir"]

    # A pristine host-side copy of the source survives even though the in-container
    # clone source is removed — the host dataset repo the harness bundles from.
    host_src = tmp_path / "host_pristine"
    subprocess.run(["cp", "-r", str(src), str(host_src)], check=True)

    # Lock down (removes the in-container clone source).
    _run_bash(
        leak_scrub.pre_agent_lockdown_script(str(testbed), base_commit, test_commit, str(src))
    )
    assert not src.exists()

    # Agent edits the solution file during its run.
    (testbed / SOLUTION_FILE).write_text("def solve():\n    return 42  # AGENT SOLUTION\n")

    # Grade time: build the bundle on the host, inject, then the grader's
    # existing stash/reset/clean/pop sequence.
    bundle = tmp_path / "grade.bundle"
    leak_scrub.build_grade_bundle(str(host_src), str(bundle), test_commit)
    inject = leak_scrub.inject_grade_bundle_script(str(testbed), str(bundle), test_commit)
    injected = _run_bash(inject)
    assert injected.returncode == 0, f"{injected.stdout}\n{injected.stderr}"
    assert "GRADE-INJECT-OK" in injected.stdout

    _git(testbed, "stash", "push", SOLUTION_FILE)
    _git(testbed, "reset", "--hard", test_commit)
    _git(testbed, "clean", "-fd")
    _git(testbed, "stash", "pop")

    # Hidden tests are back for grading, and the agent's solution is preserved.
    assert (testbed / HIDDEN_TEST_FILE).exists()
    assert HIDDEN_TEST_SECRET in (testbed / HIDDEN_TEST_FILE).read_text()
    assert "AGENT SOLUTION" in (testbed / SOLUTION_FILE).read_text()


def test_lockdown_fails_closed_when_a_stray_ref_keeps_test_commit(workspace):
    """If anything still reaches test_commit after the scrub, the verifier must
    exit nonzero rather than let the agent run against a leaky workspace."""
    testbed = workspace["testbed_dir"]
    base_commit = workspace["base_commit"]
    test_commit = workspace["test_commit"]
    src = workspace["exercise_dir"]

    # Simulate an incomplete scrub: a stray ref outside the deleted namespaces
    # (refs/keep/*) still pins test_commit and its objects.
    _git(testbed, "update-ref", "refs/keep/leak", test_commit)

    result = _run_bash(
        leak_scrub.pre_agent_lockdown_script(str(testbed), base_commit, test_commit, str(src))
    )
    assert result.returncode != 0
    assert "LOCKDOWN-FAIL" in result.stdout
