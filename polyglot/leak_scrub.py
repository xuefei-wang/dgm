"""Information-leak scrub for the polyglot agent workspace (``/testbed``).

Threat
------
``polyglot/test_spec.make_repo_script_list`` builds the agent workspace with a
local ``git clone . /testbed`` of the per-exercise repo (``/polyglot`` inside the
container) followed by ``git reset --hard {base_commit}``. Because a *local*
``git clone`` copies the entire object store, ``/testbed`` still holds
``test_commit`` (``all_files_commit`` from
``polyglot/prepare_polyglot_dataset.register_git``) reachable via ``origin/*``
refs, the local branch ref, the reflog, and loose/packed objects.
``test_commit`` contains the hidden tests AND the Exercism ``.meta/example``
reference solution. The DGM coding agent runs an unrestricted bash tool over
``/testbed``, so it can recover the graded answer offline with e.g.::

    git show origin/HEAD:<testfile>
    git cat-file -p <test_commit>:<testfile>
    git log --all ; git reflog

The ``/polyglot`` clone source is a second surface: it is the full per-exercise
tree with the hidden tests + ``.meta/example`` present on disk and in git
history, directly readable (``cat /polyglot/.../<testfile>``).

Fix (mirrors kcsi ``runtime_runner/src/workspace.ts`` ``sanitizeRepoHistory``,
issue #924)
-----------
Before the agent runs, scrub ``/testbed`` so only ``base_commit`` is reachable
(detach HEAD, drop remotes, delete every ref, expire reflogs, drop any
alternates link, ``gc --prune=now``) and remove the ``/polyglot`` clone source
entirely. Then *verify* the invariant and fail closed on any residual — a
security control must never silently no-op.

The hidden tests are re-injected at grade time from a git bundle built on the
**host** (``build_grade_bundle``), a source the in-container agent could never
read during its run. Injecting the bundle restores ``test_commit`` into the
already-scrubbed ``/testbed`` so the grader's existing
``git reset --hard {test_commit}`` works byte-for-byte as before.

This module contains only pure ``git``/shell logic (no Docker) so it is unit
testable end-to-end on a tmp repo.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

# Ref namespaces deleted during the scrub. Mirrors SANITIZE_REF_NAMESPACES in
# runtime_runner/src/workspace.ts — anything that keeps test_commit reachable
# must go. refs/heads is included so the detached HEAD is the only anchor.
_SANITIZE_REF_NAMESPACES = (
    "refs/remotes",
    "refs/tags",
    "refs/heads",
    "refs/notes",
    "refs/replace",
)

# Temporary tag used ONLY on the host to name test_commit for the grade-time
# bundle (`git bundle` refuses to bundle a bare SHA with no ref tip). The agent
# never sees it: it lives briefly on the host source repo, and — after fetch —
# in the graded /testbed, which the agent has already exited.
GRADE_BUNDLE_TAG = "kcsi-grade-testcommit"


def pre_agent_lockdown_script(
    testbed_dir: str,
    base_commit: str,
    test_commit: str,
    clone_source_dir: str,
) -> str:
    """Return a bash script that locks down the agent workspace, fail-closed.

    Scrubs ``testbed_dir`` down to only ``base_commit``, removes the
    ``clone_source_dir`` clone source, restores world-writable perms (matching
    ``make_repo_script_list``'s ``chmod -R 777``), then verifies the invariant
    and ``exit 1`` on any residual leak. Intended to run as ``root`` (removing
    the root-owned ``/polyglot`` and gc-ing require it) via
    ``container.exec_run(["/bin/bash", "-c", script], user="root")``.

    ``test_commit`` is passed only so the verifier can assert it is truly
    unrecoverable after the scrub.
    """
    testbed = shlex.quote(testbed_dir)
    base = shlex.quote(base_commit)
    test = shlex.quote(test_commit)
    src = shlex.quote(clone_source_dir)
    namespaces = " ".join(_SANITIZE_REF_NAMESPACES)
    return f"""set -u
testbed={testbed}
base={base}
test={test}
src={src}

git -C "$testbed" checkout -q --detach HEAD 2>/dev/null || true
# Drop every remote (kills `git show origin/...` and `git fetch origin <sha>`).
for r in $(git -C "$testbed" remote 2>/dev/null); do
  git -C "$testbed" remote remove "$r" 2>/dev/null || true
done
# Delete every ref; the detached HEAD keeps base_commit reachable.
git -C "$testbed" for-each-ref --format='delete %(refname)' {namespaces} 2>/dev/null \
  | git -C "$testbed" update-ref --stdin 2>/dev/null || true
# Expire reflogs so they cannot keep test_commit reachable.
git -C "$testbed" reflog expire --expire=now --all 2>/dev/null || true
# A `--shared` clone borrows objects via an alternates link that gc cannot
# prune; absorb reachable objects locally then drop the link.
if [ -f "$testbed/.git/objects/info/alternates" ]; then
  git -C "$testbed" repack -a -d -q 2>/dev/null || true
  rm -f "$testbed/.git/objects/info/alternates" 2>/dev/null || true
fi
# Physically prune the now-unreachable test_commit objects.
git -C "$testbed" gc --prune=now --quiet 2>/dev/null || true

# Remove the clone source entirely — it keeps the hidden tests + .meta example
# on disk and in history. The agent never needs it (it works in $testbed).
rm -rf "$src" 2>/dev/null || true

# Restore world-writable perms so the (possibly nonroot) agent + grader can
# operate on anything this root-run gc touched, matching setup_repo.sh.
chmod -R 777 "$testbed" 2>/dev/null || true

# ---- fail-closed verification (mirror verifySanitized in workspace.ts) ----
head=$(git -C "$testbed" rev-parse HEAD 2>/dev/null || true)
if [ "$head" != "$base" ]; then
  echo "LOCKDOWN-FAIL: HEAD ($head) != base_commit ($base)"; exit 1
fi
remotes=$(git -C "$testbed" remote 2>/dev/null || true)
if [ -n "$remotes" ]; then
  echo "LOCKDOWN-FAIL: remotes still present: $remotes"; exit 1
fi
refs=$(git -C "$testbed" for-each-ref --format='%(refname)' {namespaces} 2>/dev/null || true)
if [ -n "$refs" ]; then
  echo "LOCKDOWN-FAIL: refs still present: $refs"; exit 1
fi
beyond=$(git -C "$testbed" rev-list --all --not HEAD --count 2>/dev/null || echo MISSING)
if [ "$beyond" != "0" ]; then
  echo "LOCKDOWN-FAIL: commits reachable beyond base history: $beyond"; exit 1
fi
if git -C "$testbed" cat-file -e "$test^{{commit}}" 2>/dev/null; then
  echo "LOCKDOWN-FAIL: test_commit still recoverable in /testbed"; exit 1
fi
if git -C "$testbed" fsck --unreachable --no-dangling 2>/dev/null | grep -q '^unreachable '; then
  echo "LOCKDOWN-FAIL: unreachable objects survived gc"; exit 1
fi
if [ -f "$testbed/.git/objects/info/alternates" ]; then
  echo "LOCKDOWN-FAIL: alternates link still present"; exit 1
fi
if [ -e "$src" ]; then
  echo "LOCKDOWN-FAIL: clone source $src still present"; exit 1
fi
echo "LOCKDOWN-OK"
exit 0
"""


def build_grade_bundle(src_repo: str, bundle_path: str, test_commit: str) -> None:
    """Build, on the host, a git bundle carrying ``test_commit`` and ancestors.

    ``src_repo`` is the pristine per-exercise repo on the host (the dataset
    ``repo`` field), whose history contains ``test_commit``. Tags the commit so
    ``git bundle`` has a ref tip (it refuses a bare SHA), bundles the tag, then
    removes the tag. Raises ``CalledProcessError`` on failure so the caller can
    fail closed.
    """
    src_repo = str(src_repo)
    bundle_path = str(bundle_path)
    subprocess.run(
        ["git", "-C", src_repo, "tag", "-f", GRADE_BUNDLE_TAG, test_commit],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        subprocess.run(
            ["git", "-C", src_repo, "bundle", "create", bundle_path, f"refs/tags/{GRADE_BUNDLE_TAG}"],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        subprocess.run(
            ["git", "-C", src_repo, "tag", "-d", GRADE_BUNDLE_TAG],
            check=False,
            capture_output=True,
            text=True,
        )


def inject_grade_bundle_script(
    testbed_dir: str,
    bundle_path: str,
    test_commit: str,
) -> str:
    """Return a bash script that fetches ``test_commit`` from a host-built bundle.

    Runs inside the container at grade time (after the agent has exited) to
    restore ``test_commit`` into the scrubbed ``testbed_dir`` so the grader's
    ``git reset --hard {test_commit}`` works. Fail-closed: ``exit 1`` if the
    commit is not present after the fetch.
    """
    testbed = shlex.quote(testbed_dir)
    bundle = shlex.quote(bundle_path)
    test = shlex.quote(test_commit)
    return f"""set -u
git -C {testbed} fetch -q {bundle} \
  'refs/tags/{GRADE_BUNDLE_TAG}:refs/tags/{GRADE_BUNDLE_TAG}' 2>/dev/null || true
if ! git -C {testbed} cat-file -e {test}'^{{commit}}' 2>/dev/null; then
  echo "GRADE-INJECT-FAIL: test_commit not present after bundle fetch"; exit 1
fi
echo "GRADE-INJECT-OK"
exit 0
"""
