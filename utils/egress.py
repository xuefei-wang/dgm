"""Network egress isolation for the DGM self-improvement container.

Design mirrors KCSI's egress isolation (parent repo
``runtime_runner/src/container_args.ts`` ``ensureEgressInfra`` +
``egress_allowlist.ts``): the self-improvement agent container is attached to a
Docker ``--internal`` network with no route to the internet; its only egress is
an allowlisting HTTP CONNECT proxy sidecar (``egress_proxy.py``) that permits
only the configured provider API (and PyPI, which the self-improve step
pip-installs from). This closes the arbitrary-host channel a meta-agent with
unrestricted bash could use to re-fetch public hidden tests.

Default is ISOLATED. Set ``KCSI_DGM_EGRESS_OPEN`` truthy (1/true/yes/on) to
restore the legacy open-bridge behaviour for debugging. Extra allowlist hosts
can be added with ``KCSI_DGM_EGRESS_ALLOW=host1,host2``.

The pure helpers (``egress_open``, ``derive_allowlist``, ``agent_run_kwargs``,
``proxy_run_kwargs``) are unit-tested without a Docker daemon; the lifecycle
(``ensure_egress_infra`` / ``teardown_egress_infra``) requires a live daemon.
"""
import os
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

PROXY_PORT = 8080
_TRUTHY = {"1", "true", "yes", "on"}

# PyPI is allowlisted because self_improve_step re-runs
# ``pip install -r requirements.txt`` inside the container (self_improve_step.py
# ~line 543); it is not a hidden-test fetch channel. github.com / exercism and
# all other hosts stay blocked.
_PYPI_HOSTS = ["pypi.org", "files.pythonhosted.org"]


def egress_open(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get("KCSI_DGM_EGRESS_OPEN", "")).strip().lower() in _TRUTHY


def _add_url_host(hosts: set, raw) -> None:
    if not raw or not str(raw).strip():
        return
    try:
        host = urlparse(str(raw).strip()).hostname
    except ValueError:
        host = None
    if host:
        hosts.add(host.lower())


def derive_allowlist(env=None) -> list:
    """Provider API hosts (from present credentials) + base-URL hosts + PyPI +
    operator extras. Falls back to the anthropic+openai defaults if no provider
    credential is detected, so isolation never silently blocks all model calls.
    """
    env = os.environ if env is None else env
    hosts = set(_PYPI_HOSTS)

    if env.get("OPENAI_API_KEY"):
        hosts.add("api.openai.com")
        _add_url_host(hosts, env.get("OPENAI_BASE_URL"))
        _add_url_host(hosts, env.get("OPENAI_API_BASE"))
    if env.get("ANTHROPIC_API_KEY"):
        hosts.add("api.anthropic.com")
        _add_url_host(hosts, env.get("ANTHROPIC_BASE_URL"))
    if env.get("AWS_ACCESS_KEY_ID") or env.get("AWS_SECRET_ACCESS_KEY"):
        region = (
            env.get("AWS_REGION")
            or env.get("AWS_REGION_NAME")
            or env.get("AWS_DEFAULT_REGION")
            or "us-east-1"
        )
        hosts.add(f"bedrock-runtime.{region}.amazonaws.com")
        hosts.add(f"bedrock.{region}.amazonaws.com")
        hosts.add("sts.amazonaws.com")
        hosts.add(f"sts.{region}.amazonaws.com")
    if env.get("OPENROUTER_API_KEY"):
        hosts.add("openrouter.ai")
    if env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY"):
        hosts.add("generativelanguage.googleapis.com")
    if env.get("DEEPSEEK_API_KEY"):
        hosts.add("api.deepseek.com")

    # If no provider credential was present, keep the two common defaults so a
    # misconfigured-but-isolated run still reaches a provider rather than 403ing
    # every call.
    if not (hosts - set(_PYPI_HOSTS)):
        hosts.add("api.anthropic.com")
        hosts.add("api.openai.com")

    for raw in str(env.get("KCSI_DGM_EGRESS_ALLOW", "")).split(","):
        host = raw.strip().lower()
        if host:
            hosts.add(host)

    return sorted(hosts)


@dataclass
class EgressInfra:
    internal_net: str
    external_net: str
    proxy_name: str
    proxy_port: int


def _proxy_env(infra: EgressInfra) -> dict:
    url = f"http://{infra.proxy_name}:{infra.proxy_port}"
    return {
        "HTTPS_PROXY": url,
        "HTTP_PROXY": url,
        "https_proxy": url,
        "http_proxy": url,
        "NO_PROXY": "localhost,127.0.0.1",
        "no_proxy": "localhost,127.0.0.1",
    }


def agent_run_kwargs(infra) -> dict:
    """docker-SDK ``containers.run`` kwargs that attach the agent container to
    the internal (no-route) network and point it at the proxy. Empty dict in
    open mode (``infra is None``)."""
    if infra is None:
        return {}
    return {
        "network": infra.internal_net,
        "environment": _proxy_env(infra),
        # Blackhole external DNS (mirror KCSI #934): the embedded resolver still
        # answers the proxy container name locally; the proxy resolves provider
        # hosts on the agent's behalf, so the agent never needs external DNS.
        "dns": ["0.0.0.0"],
    }


def proxy_run_kwargs(image_name: str, infra: EgressInfra, allowlist: list, script_path: str) -> dict:
    """docker-SDK ``containers.run`` kwargs for the proxy sidecar."""
    return {
        "image": image_name,
        "name": infra.proxy_name,
        "detach": True,
        "network": infra.external_net,
        "command": ["python", "/egress_proxy.py"],
        "entrypoint": "",
        "environment": {
            "KCSI_EGRESS_PROXY_PORT": str(infra.proxy_port),
            "KCSI_EGRESS_ALLOWLIST": ",".join(allowlist),
        },
        "volumes": {script_path: {"bind": "/egress_proxy.py", "mode": "ro"}},
    }


def ensure_egress_infra(client, image_name: str, run_id: str, env=None):
    """Create the internal+external networks and the allowlisting proxy sidecar.

    Returns an ``EgressInfra`` (isolated) or ``None`` (open mode). Fails CLOSED:
    any setup error tears down partial infra and raises rather than silently
    running unisolated. Requires a live Docker daemon.
    """
    if egress_open(env):
        return None

    allowlist = derive_allowlist(env)
    infra = EgressInfra(
        internal_net=f"dgm-egress-int-{run_id}",
        external_net=f"dgm-egress-ext-{run_id}",
        proxy_name=f"dgm-egress-proxy-{run_id}",
        proxy_port=PROXY_PORT,
    )
    script_path = str(Path(__file__).resolve().parent / "egress_proxy.py")

    created = []
    proxy = None
    try:
        int_net = client.networks.create(infra.internal_net, driver="bridge", internal=True)
        created.append(int_net)
        ext_net = client.networks.create(infra.external_net, driver="bridge", internal=False)
        created.append(ext_net)

        proxy = client.containers.run(
            **proxy_run_kwargs(image_name, infra, allowlist, script_path)
        )
        int_net.connect(proxy)

        if not _wait_proxy_ready(proxy, timeout=20):
            raise RuntimeError("egress proxy did not become READY within 20s")
        return infra
    except Exception:
        _teardown(client, infra, proxy, created)
        raise


def _wait_proxy_ready(proxy, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            logs = proxy.logs().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            logs = ""
        if "[egress-proxy] READY" in logs:
            return True
        time.sleep(0.3)
    return False


def _teardown(client, infra, proxy, networks) -> None:
    if proxy is not None:
        try:
            proxy.remove(force=True)
        except Exception:  # noqa: BLE001
            pass
    else:
        try:
            client.containers.get(infra.proxy_name).remove(force=True)
        except Exception:  # noqa: BLE001
            pass
    for net in networks:
        try:
            net.remove()
        except Exception:  # noqa: BLE001
            pass


def teardown_egress_infra(client, infra) -> None:
    """Best-effort teardown of the proxy container and both networks."""
    if infra is None:
        return
    try:
        client.containers.get(infra.proxy_name).remove(force=True)
    except Exception:  # noqa: BLE001
        pass
    for name in (infra.internal_net, infra.external_net):
        try:
            client.networks.get(name).remove()
        except Exception:  # noqa: BLE001
            pass
