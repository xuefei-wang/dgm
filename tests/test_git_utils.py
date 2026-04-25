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
