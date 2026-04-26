import subprocess
from pathlib import Path

from utils.git_utils import diff_versus_commit


def _run(cmd, cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def test_diff_versus_commit_ignores_generated_untracked_artifacts(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init"], cwd=repo)
    _run(["git", "config", "user.name", "Test User"], cwd=repo)
    _run(["git", "config", "user.email", "test@example.com"], cwd=repo)

    tracked = repo / "main.py"
    tracked.write_text("print('before')\n", encoding="utf-8")
    _run(["git", "add", "main.py"], cwd=repo)
    _run(["git", "commit", "-m", "init"], cwd=repo)
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()

    tracked.write_text("print('after')\n", encoding="utf-8")
    (repo / "notes.txt").write_text("keep me\n", encoding="utf-8")
    (repo / "target").mkdir()
    (repo / "target" / "debug.log").write_text("generated\n", encoding="utf-8")
    (repo / "appendonlydir").mkdir()
    (repo / "appendonlydir" / "appendonly.aof").write_text("generated\n", encoding="utf-8")
    (repo / "Cargo.lock").write_text("generated\n", encoding="utf-8")
    (repo / "dump.rdb").write_text("generated\n", encoding="utf-8")

    diff = diff_versus_commit(str(repo), base_commit)

    assert "main.py" in diff
    assert "notes.txt" in diff
    assert "target/debug.log" not in diff
    assert "appendonlydir/appendonly.aof" not in diff
    assert "Cargo.lock" not in diff
    assert "dump.rdb" not in diff


def test_diff_versus_commit_keeps_tracked_files_with_ignored_names(tmp_path: Path) -> None:
    """Filtering applies only to untracked files; tracked Cargo.lock /
    dump.rdb modifications must still appear in the patch."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init"], cwd=repo)
    _run(["git", "config", "user.name", "Test User"], cwd=repo)
    _run(["git", "config", "user.email", "test@example.com"], cwd=repo)

    cargo_lock = repo / "Cargo.lock"
    cargo_lock.write_text("version = 3\n", encoding="utf-8")
    dump_rdb = repo / "dump.rdb"
    dump_rdb.write_text("rdb-v1\n", encoding="utf-8")
    nested = repo / "build"
    nested.mkdir()
    nested_file = nested / "intentional.txt"
    nested_file.write_text("tracked under build/\n", encoding="utf-8")
    _run(["git", "add", "Cargo.lock", "dump.rdb", "build/intentional.txt"], cwd=repo)
    _run(["git", "commit", "-m", "init"], cwd=repo)
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()

    cargo_lock.write_text("version = 4\n", encoding="utf-8")
    dump_rdb.write_text("rdb-v2\n", encoding="utf-8")
    nested_file.write_text("tracked under build/ updated\n", encoding="utf-8")

    diff = diff_versus_commit(str(repo), base_commit)

    assert "Cargo.lock" in diff
    assert "dump.rdb" in diff
    assert "build/intentional.txt" in diff


def test_diff_versus_commit_filters_deeply_nested_artifacts(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init"], cwd=repo)
    _run(["git", "config", "user.name", "Test User"], cwd=repo)
    _run(["git", "config", "user.email", "test@example.com"], cwd=repo)
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _run(["git", "add", "seed.txt"], cwd=repo)
    _run(["git", "commit", "-m", "init"], cwd=repo)
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()

    nested = repo / "pkg" / "sub" / "__pycache__"
    nested.mkdir(parents=True)
    (nested / "mod.cpython-312.pyc").write_text("bytecode\n", encoding="utf-8")
    (repo / "src" / "main" / "build").mkdir(parents=True)
    (repo / "src" / "main" / "build" / "out.o").write_text("obj\n", encoding="utf-8")

    diff = diff_versus_commit(str(repo), base_commit)

    assert "__pycache__" not in diff
    assert "build/out.o" not in diff
