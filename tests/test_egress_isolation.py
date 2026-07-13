"""Unit tests for DGM self-improvement container egress isolation.

Daemon-free: exercises the pure allowlist/mode/kwargs helpers, a real
localhost round-trip through the CONNECT proxy (allow vs deny), and that
``build_dgm_container`` threads the isolation kwargs into ``containers.run``.
"""
import socket
import sys
import threading
import time
import types

import pytest

# Stub the `docker` package if it isn't installed in the test env (mirrors the
# other DGM tests) so importing utils.docker_utils doesn't fail at collection.
try:
    import docker  # noqa: F401
except ModuleNotFoundError:
    docker_module = types.ModuleType("docker")
    docker_errors = types.ModuleType("docker.errors")
    docker_errors.NotFound = type("NotFound", (Exception,), {})
    docker_errors.APIError = type("APIError", (Exception,), {})
    docker_module.errors = docker_errors
    sys.modules.setdefault("docker", docker_module)
    sys.modules.setdefault("docker.errors", docker_errors)

from utils import egress
from utils import egress_proxy


# --- pure helpers ---------------------------------------------------------

def test_egress_open_default_is_isolated():
    assert egress.egress_open({}) is False


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
def test_egress_open_truthy(val):
    assert egress.egress_open({"KCSI_DGM_EGRESS_OPEN": val}) is True


@pytest.mark.parametrize("val", ["", "0", "false", "no", "off", "nope"])
def test_egress_open_falsy(val):
    assert egress.egress_open({"KCSI_DGM_EGRESS_OPEN": val}) is False


def test_derive_allowlist_always_includes_pypi():
    allow = egress.derive_allowlist({"ANTHROPIC_API_KEY": "x"})
    assert "pypi.org" in allow
    assert "files.pythonhosted.org" in allow


def test_derive_allowlist_provider_hosts_from_creds():
    assert "api.anthropic.com" in egress.derive_allowlist({"ANTHROPIC_API_KEY": "x"})
    assert "api.openai.com" in egress.derive_allowlist({"OPENAI_API_KEY": "x"})
    assert "api.openai.com" not in egress.derive_allowlist({"ANTHROPIC_API_KEY": "x"})


def test_derive_allowlist_never_includes_github():
    allow = egress.derive_allowlist({"ANTHROPIC_API_KEY": "x", "OPENAI_API_KEY": "y"})
    assert "github.com" not in allow


def test_derive_allowlist_bedrock_region():
    allow = egress.derive_allowlist({"AWS_ACCESS_KEY_ID": "x", "AWS_REGION": "eu-west-1"})
    assert "bedrock-runtime.eu-west-1.amazonaws.com" in allow


def test_derive_allowlist_base_url_host():
    allow = egress.derive_allowlist(
        {"OPENAI_API_KEY": "x", "OPENAI_BASE_URL": "https://gateway.example.com/v1"}
    )
    assert "gateway.example.com" in allow


def test_derive_allowlist_extra_hosts():
    allow = egress.derive_allowlist({"ANTHROPIC_API_KEY": "x", "KCSI_DGM_EGRESS_ALLOW": "a.com, b.com"})
    assert "a.com" in allow and "b.com" in allow


def test_derive_allowlist_fallback_when_no_creds():
    allow = egress.derive_allowlist({})
    assert "api.anthropic.com" in allow and "api.openai.com" in allow


def test_agent_run_kwargs_open_is_empty():
    assert egress.agent_run_kwargs(None) == {}


def test_agent_run_kwargs_isolated():
    infra = egress.EgressInfra("int", "ext", "proxyhost", 8080)
    kw = egress.agent_run_kwargs(infra)
    assert kw["network"] == "int"
    assert kw["dns"] == ["0.0.0.0"]
    assert kw["environment"]["HTTPS_PROXY"] == "http://proxyhost:8080"
    assert kw["environment"]["http_proxy"] == "http://proxyhost:8080"


def test_proxy_run_kwargs_shape():
    infra = egress.EgressInfra("int", "ext", "proxyhost", 8080)
    kw = egress.proxy_run_kwargs("dgm", infra, ["api.anthropic.com"], "/host/egress_proxy.py")
    assert kw["image"] == "dgm"
    assert kw["network"] == "ext"
    assert kw["command"] == ["python", "/egress_proxy.py"]
    assert kw["entrypoint"] == ""
    assert kw["environment"]["KCSI_EGRESS_ALLOWLIST"] == "api.anthropic.com"
    assert kw["volumes"]["/host/egress_proxy.py"]["bind"] == "/egress_proxy.py"


# --- proxy allowlist logic ------------------------------------------------

def test_host_allowed_exact_and_subdomain():
    allow = {"api.anthropic.com", "amazonaws.com"}
    assert egress_proxy.host_allowed("api.anthropic.com", allow)
    assert egress_proxy.host_allowed("bedrock.us-east-1.amazonaws.com", allow)
    assert not egress_proxy.host_allowed("github.com", allow)
    assert not egress_proxy.host_allowed("evil-api.anthropic.com.attacker.net", allow)


# --- real CONNECT round-trip through the proxy (no Docker) -----------------

def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_echo_server(port):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)

    def serve():
        try:
            conn, _ = srv.accept()
            data = conn.recv(1024)
            conn.sendall(data)
            conn.close()
        except OSError:
            pass

    threading.Thread(target=serve, daemon=True).start()
    return srv


def test_proxy_connect_allowed_and_denied(monkeypatch):
    proxy_port = _free_port()
    echo_port = _free_port()
    monkeypatch.setattr(egress_proxy, "PORT", proxy_port)
    # Allow "localhost" so the CONNECT target host resolves to our echo server.
    monkeypatch.setattr(egress_proxy, "ALLOW", {"localhost"})

    server_thread = threading.Thread(target=egress_proxy.main, daemon=True)
    server_thread.start()
    time.sleep(0.3)

    # DENY: a host not on the allowlist -> 403
    c = socket.create_connection(("127.0.0.1", proxy_port), timeout=5)
    c.sendall(b"CONNECT github.com:443 HTTP/1.1\r\nHost: github.com\r\n\r\n")
    resp = c.recv(1024)
    c.close()
    assert b"403" in resp

    # ALLOW: localhost tunnel to the echo server, then bytes flow through
    echo = _start_echo_server(echo_port)
    c = socket.create_connection(("127.0.0.1", proxy_port), timeout=5)
    c.sendall(
        f"CONNECT localhost:{echo_port} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode()
    )
    resp = c.recv(1024)
    assert b"200" in resp
    c.sendall(b"ping")
    assert c.recv(1024) == b"ping"
    c.close()
    echo.close()


# --- build_dgm_container wiring -------------------------------------------

class _FakeContainer:
    def start(self):
        pass


class _FakeContainers:
    def __init__(self):
        self.run_kwargs = None

    def run(self, **kwargs):
        self.run_kwargs = kwargs
        return _FakeContainer()


class _FakeImages:
    def list(self):
        # Pretend the image already exists so no build is attempted.
        img = types.SimpleNamespace(tags=["dgm"])
        return [img]

    def build(self, **kwargs):
        return (types.SimpleNamespace(tags=["dgm"]), [])


class _FakeClient:
    def __init__(self):
        self.containers = _FakeContainers()
        self.images = _FakeImages()


def test_build_dgm_container_isolated_attaches_internal_network():
    from utils.docker_utils import build_dgm_container, setup_logger
    setup_logger("/tmp/dgm_egress_test.log")
    client = _FakeClient()
    infra = egress.EgressInfra("dgm-egress-int-1", "dgm-egress-ext-1", "dgm-egress-proxy-1", 8080)
    build_dgm_container(client, "./", "dgm", "dgm-container-1", egress=infra)
    kw = client.containers.run_kwargs
    assert kw["network"] == "dgm-egress-int-1"
    assert kw["dns"] == ["0.0.0.0"]
    assert kw["environment"]["HTTPS_PROXY"] == "http://dgm-egress-proxy-1:8080"
    assert "network_mode" not in kw


def test_build_dgm_container_open_has_no_network_override():
    from utils.docker_utils import build_dgm_container, setup_logger
    setup_logger("/tmp/dgm_egress_test.log")
    client = _FakeClient()
    build_dgm_container(client, "./", "dgm", "dgm-container-2", egress=None)
    kw = client.containers.run_kwargs
    assert "network" not in kw
    assert "environment" not in kw
