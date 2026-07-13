#!/usr/bin/env python3
"""Egress-isolation smoke test for the DGM self-improvement container.

Analogous to KCSI's ``scripts/experiments/egress_smoke.sh``. Proves that, with
default-isolated egress, a blocked host (github.com) is unreachable from inside
the container while the provider API (api.anthropic.com) is reachable through
the allowlisting proxy sidecar.

REQUIRES a live Docker daemon AND the ``dgm`` image already built
(``docker build -t dgm .`` from baselines/dgm). Run:

    python egress_smoke.py

Exit 0 = isolation working (github blocked, provider reachable); non-zero
otherwise. Set KCSI_DGM_EGRESS_OPEN=1 to observe the legacy open behaviour
(both reachable) for contrast.
"""
import sys
import uuid

import docker

from utils.egress import (
    agent_run_kwargs,
    egress_open,
    ensure_egress_infra,
    teardown_egress_infra,
)

IMAGE = "dgm"
BLOCKED = "https://github.com"
ALLOWED = "https://api.anthropic.com"  # 401 without a key still proves reachability


# Probe with python3 (always present in the image) rather than curl, which is
# NOT installed in python:*-slim bases. urllib honors the HTTPS_PROXY that
# agent_run_kwargs injects, so it exercises the same egress path a real agent
# would. Classification: a server response (200, or 401/404 surfaced as
# HTTPError) means the host was REACHED through the proxy; a tunnel/connection
# failure (the allowlist proxy answering 403 to CONNECT, or no route) is BLOCKED.
_PROBE_PY = (
    "import sys, urllib.request, urllib.error\n"
    "try:\n"
    "    r = urllib.request.urlopen(sys.argv[1], timeout=12)\n"
    "    print('REACHED', r.status)\n"
    "except urllib.error.HTTPError as e:\n"
    "    print('REACHED', e.code)\n"
    "except Exception as e:\n"
    "    print('BLOCKED', type(e).__name__, str(getattr(e, 'reason', e))[:60])\n"
)


def _probe(container, url):
    res = container.exec_run(["python3", "-c", _PROBE_PY, url])
    return res.output.decode().strip()


def main():
    client = docker.from_env()
    run_id = f"smoke-{uuid.uuid4().hex[:8]}"
    infra = ensure_egress_infra(client, IMAGE, run_id)
    if infra is None:
        print("KCSI_DGM_EGRESS_OPEN is set: egress isolation DISABLED (open mode).")

    kwargs = dict(image=IMAGE, name=f"dgm-egress-smoke-{run_id}", detach=True,
                  command="tail -f /dev/null")
    kwargs.update(agent_run_kwargs(infra))
    container = client.containers.run(**kwargs)
    try:
        blocked = _probe(container, BLOCKED)
        allowed = _probe(container, ALLOWED)
        print(f"{BLOCKED} -> {blocked}")
        print(f"{ALLOWED} -> {allowed}")

        if egress_open():
            print("open mode: no assertions.")
            return 0

        ok = blocked.startswith("BLOCKED") and allowed.startswith("REACHED")
        if ok:
            print("PASS: blocked host unreachable, provider reachable via proxy.")
            return 0
        print("FAIL: isolation did not behave as expected.")
        return 1
    finally:
        try:
            container.remove(force=True)
        except Exception:  # noqa: BLE001
            pass
        teardown_egress_infra(client, infra)


if __name__ == "__main__":
    sys.exit(main())
