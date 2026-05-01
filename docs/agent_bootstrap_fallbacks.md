# SWE-bench Pro agent bootstrap: fallback chain for legacy base images

## Problem

`pro_harness.py:_setup_agent_python` provisions the Python environment that
runs the DGM coding agent (anthropic + openai SDKs) inside each per-instance
SWE-bench Pro Docker container. The upstream implementation tried only two
strategies and failed silently when both gapped at the same time, blocking
DGM from running on tasks whose base image happens to be older.

Concretely, in the SWE-bench Pro 50-task seed-0 audit subset
(`swebench_pro_test_50_seed0_v1.json`), one of the 10 audit instances
(`instance_gravitational__teleport-eda668c30d9d3b56d9c69197b120b01013611186`)
ships an older Debian-bullseye base image whose Python toolchain trips three
constraints at once:

| Image attribute | 9 healthy instances | teleport-eda |
|---|---|---|
| Distro | Debian 12 (bookworm) | Debian 11 (bullseye) |
| System Python | 3.11.2 | 3.9.2 |
| `python3-venv` package | preinstalled | not installed; not in apt cache |
| pip version | ≥ 23 (PEP 668-aware) | 20.3.4 |

DGM's `llm.py` line 162 uses PEP 604 union syntax:

```python
def _bedrock_region() -> str | None:
```

which requires Python ≥ 3.10. On the bullseye base image, this raises
`TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'` at
import time, before any agent reasoning begins.

### Original fallback chain (gap)

```
1. python3 -m venv /dgm/.venv      # fails: no python3-venv → no ensurepip
2. pip install --break-system-packages -r requirements.txt
                                    # fails: pip 20.3.4 doesn't recognise the flag
                                    # ("no such option: --break-system-packages")
   → return python_bin (no agent deps actually installed)
```

When both strategies fail the function returns the system Python anyway. The
agent then crashes on `import anthropic` (and even if anthropic were present,
on the PEP 604 syntax in `llm.py`). The container exits with status 2, no
agent log is produced, and `make_pro_initial_metadata.py` refuses to package
the initial state because one task's prediction artefacts are missing —
blocking the entire `DGM_outer.py --swebench_pro` outer loop from starting.

### Observed failure (DGM swarms-integration branch, commit 1e8263d)

```
File "/dgm/llm.py", line 162, in <module>
    def _bedrock_region() -> str | None:
TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
```

Plus, slightly earlier in the same docker.log:

```
The virtual environment was not created successfully because ensurepip is not
available.  On Debian/Ubuntu systems, you need to install the python3-venv
package using the following command.

    apt-get install python3-venv

Failing command: ['/dgm/.venv/bin/python3', '-Im', 'ensurepip', '--upgrade', '--default-pip']

# ... later ...

no such option: --break-system-packages
```

## Fix

`_setup_agent_python` is extended from 2 to 4 strategies, ordered so existing
images keep their existing path:

```
0. (NEW) If container Python < 3.10:
     install uv → uv python install 3.11 → /dgm/.venv on uv-managed Python 3.11
1. python3 -m venv (existing)
1b. (NEW) If venv fails:
     apt-get install python3-venv (with apt-get update fallback) → retry venv
2. pip install --break-system-packages (existing)
3. (NEW) bare pip install -r requirements.txt
     for legacy pip < 23 that pre-dates PEP 668 enforcement
```

Strategies 1, 2, and 3 share the same install command builder
(`_agent_pip_install_command`). Strategy 0 lifts the entire stack onto a
known-good Python 3.11 build before anything else runs, which is necessary
when the container's system Python is too old for DGM source itself, not
just for SDK installation.

### Diff

```
swe_bench/pro_harness.py   | 47 lines added, 8 lines refactored
```

The function preserves all upstream behaviour for healthy images:

- Containers with Python ≥ 3.10 and python3-venv preinstalled hit the same
  Strategy 1 path as before; Strategies 0, 1b, 2, 3 do not execute.
- Containers with Python ≥ 3.10 but missing python3-venv now succeed via
  Strategy 1b instead of returning a broken environment.
- Containers with Python < 3.10 (the teleport-eda case) now succeed via
  Strategy 0 with a uv-managed Python 3.11 venv. Agent still imports the same
  `anthropic 0.97.0`, `openai 2.33.0`, etc., from the same `requirements.txt`,
  with the same model and the same prompts.

### Why uv

uv (`https://astral.sh/uv`) ships statically-linked
[python-build-standalone](https://github.com/astral-sh/python-build-standalone)
Python interpreters that work on any Linux without touching system apt.
Tested install in the failing image: uv binary downloads in <1s,
`uv python install 3.11` finishes in ~500ms, total Strategy 0 wall-clock
overhead is roughly 2-4 seconds amortised over a multi-minute agent run.

The alternative paths considered and rejected:

- **bullseye-backports**: Debian 11 backports no longer maintained;
  `apt-cache search python3.11` returns no candidate.
- **deadsnakes PPA**: Ubuntu-targeted, fails on Debian.
- **Patching `llm.py` to use `Optional[str]`**: would change the agent code
  under test. The DGM authors set Python ≥ 3.10 as a requirement; the right
  fix is environmental, not source.
- **Building Python from source**: too slow for a per-task bootstrap.

## Verification

End-to-end smoke test inside the actual failing image:

```bash
docker run --rm \
  -v /path/to/dgm:/dgm-host-mount:ro \
  --entrypoint /bin/bash \
  jefzda/sweap-images:gravitational.teleport-gravitational__teleport-eda668c30d9d3b56d9c69197b120b01013611186 \
  -c '
    # Simulate Strategy 0 chain
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/root/.local/bin sh -s -- --quiet
    /root/.local/bin/uv python install 3.11
    PY=$(/root/.local/bin/uv python find 3.11)
    $PY -m venv /dgm/.venv
    /dgm/.venv/bin/python -m pip install --quiet --index-url https://pypi.org/simple \
      -r /dgm-host-mount/requirements.txt

    # Confirm DGM modules import (the original failure was here)
    cp /dgm-host-mount/llm.py /dgm-host-mount/llm_withtools.py /dgm/
    cd /dgm && /dgm/.venv/bin/python -c "
import sys; sys.path.insert(0, \"/dgm\")
import llm; print(\"llm OK\")
from llm_withtools import CLAUDE_MODEL
print(\"llm_withtools OK\")
    "
  '
```

Expected output (verified):

```
Installed Python 3.11.15 in 504ms
Successfully installed anthropic-0.97.0 openai-2.33.0 ...
llm OK
llm_withtools OK
```

In the live audit run after deploying this patch:

- Strategy 0 fired exactly on teleport-eda, exactly as designed.
- Agent successfully made 67+ Haiku 4.5 LLM round-trips with normal token
  growth (~52K input / ~100 output per turn) — same shape as the 9 healthy
  instances.
- pip install in the venv produced byte-identical SDK versions
  (anthropic 0.97.0, openai 2.33.0) to the 9 healthy instances.

## Honesty and reproducibility

- Strategies 1-3 are byte-identical to upstream's flow on images that don't
  need them. The 9 healthy instances are completely unaffected.
- Strategy 0 only activates when `container_minor < 10`. On healthy images
  this branch never executes.
- The agent under test (`coding_agent.py`, `llm.py`, `tools/`, `prompts/`,
  model selection, scoring) is unchanged — this patch only fixes the runtime
  bootstrap so the agent can actually run.
- Cross-image consistency for legacy-base tasks: with this patch, those
  tasks run on uv-managed CPython 3.11.x; healthy tasks run on system
  CPython 3.11.x. Both are Python 3.11 patch series with no API/behaviour
  divergence relevant to LLM-driven coding agents.

## Reproducing the original failure (without the patch)

```bash
git -C baselines/dgm checkout 1e8263dd236dcc2855528d80b784aa465fee6d80
cd baselines/dgm
env UV_PROJECT_ENVIRONMENT=$BASELINE_VENV_DIR uv run python swe_bench/pro_harness.py \
  --dataset-path /path/to/swebench_pro/dataset/test.jsonl \
  --task-map /path/to/audit_seed0_subset.json \
  --pred-dname /tmp/pred --output-dir /tmp/reports \
  --model-name reproducer --max-workers 1
# expect: instance_gravitational__teleport-eda... fails with
#   "Script failed with exit code 2"
# and downstream:
#   FileNotFoundError: Missing prediction JSON, agent log, or eval log
```

## Files changed

- `swe_bench/pro_harness.py` — `_setup_agent_python` extended

No new dependencies, no new tests required (function is exercised end-to-end
by every SWE-bench Pro run).

## Upstream PR plan

1. **`xuefei-wang/dgm`** (this repo, `swarms-integration` branch):
   - Open PR with this commit + this doc as PR description.
   - Targets `swarms-integration` since the swarms-side workflow uses that
     branch as its baseline pin.

2. **`xuefei-wang/swarms`**:
   - Once (1) lands, bump `baselines/dgm` submodule pin to the merge commit
     in a small follow-up PR.
   - No swarms-side code changes.
