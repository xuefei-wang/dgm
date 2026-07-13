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


def _curl(container, url):
    # %{http_code} is 000 when the connection fails (blocked); any non-000 code
    # (200/401/403) means the host was reached.
    cmd = ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "12", url]
    res = container.exec_run(cmd)
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
        blocked_code = _curl(container, BLOCKED)
        allowed_code = _curl(container, ALLOWED)
        print(f"{BLOCKED} -> HTTP {blocked_code}")
        print(f"{ALLOWED} -> HTTP {allowed_code}")

        if egress_open():
            print("open mode: no assertions.")
            return 0

        ok = blocked_code == "000" and allowed_code != "000"
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
