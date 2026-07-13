#!/usr/bin/env python3
"""Allowlisting HTTP CONNECT forward proxy (stdlib-only).

Mirrors KCSI's egress proxy conceptually (see the parent repo's
``runtime_runner/src/container_args.ts`` ``proxyContainerRunArgs`` /
``ensureEgressInfra`` and ``egress_allowlist.ts``): the self-improvement agent
container is attached to a Docker ``--internal`` network with NO route to the
internet, so its only egress path is via this sidecar. The sidecar permits
``CONNECT`` tunnels ONLY to allowlisted hostnames (the configured LLM provider
API plus PyPI); every other host is answered ``403 Forbidden``.

This closes the arbitrary-host egress channel (``git clone github.com`` /
``curl exercism`` / raw sockets) that a meta/self-improvement agent with
unrestricted bash could otherwise use to re-fetch public hidden tests, while
preserving the provider API access the model calls need.

Run inside the sidecar as ``python /egress_proxy.py``. Configuration is via env:
  KCSI_EGRESS_ALLOWLIST : comma-separated allowlisted hostnames
  KCSI_EGRESS_PROXY_PORT: listen port (default 8080)
"""
import os
import select
import socket
import sys
import threading

PORT = int(os.environ.get("KCSI_EGRESS_PROXY_PORT", "8080"))
ALLOW = {
    h.strip().lower().rstrip(".")
    for h in os.environ.get("KCSI_EGRESS_ALLOWLIST", "").split(",")
    if h.strip()
}


def host_allowed(host: str, allow=None) -> bool:
    """True if ``host`` matches an allowlist entry exactly or as a subdomain."""
    allow = ALLOW if allow is None else allow
    host = host.strip().lower().rstrip(".")
    if not host:
        return False
    for entry in allow:
        if host == entry or host.endswith("." + entry):
            return True
    return False


def _log(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _pipe(a: socket.socket, b: socket.socket) -> None:
    socks = [a, b]
    try:
        while True:
            readable, _, errored = select.select(socks, [], socks, 60)
            if errored or not readable:
                return
            for s in readable:
                other = b if s is a else a
                data = s.recv(65536)
                if not data:
                    return
                other.sendall(data)
    finally:
        for s in socks:
            try:
                s.close()
            except OSError:
                pass


def handle(client: socket.socket) -> None:
    try:
        client.settimeout(30)
        req = b""
        while b"\r\n\r\n" not in req:
            chunk = client.recv(4096)
            if not chunk:
                client.close()
                return
            req += chunk
            if len(req) > 65536:
                client.close()
                return
        line = req.split(b"\r\n", 1)[0].decode("latin1")
        parts = line.split()
        if len(parts) < 2 or parts[0].upper() != "CONNECT":
            # Only tunneled HTTPS (CONNECT) is supported; plain-HTTP forwarding
            # is intentionally unsupported (provider APIs are HTTPS).
            client.sendall(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
            client.close()
            return
        target = parts[1]
        host, _, port_s = target.partition(":")
        try:
            port = int(port_s) if port_s else 443
        except ValueError:
            port = 443
        if not host_allowed(host):
            _log(f"[egress-proxy] DENY {host}:{port}")
            client.sendall(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            client.close()
            return
        try:
            upstream = socket.create_connection((host, port), timeout=30)
        except OSError:
            client.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            client.close()
            return
        client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        _log(f"[egress-proxy] ALLOW {host}:{port}")
        _pipe(client, upstream)
    except Exception:  # noqa: BLE001 - never let one connection kill the proxy
        try:
            client.close()
        except OSError:
            pass


def main() -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", PORT))
    srv.listen(128)
    _log(f"[egress-proxy] READY on 0.0.0.0:{PORT} allow={sorted(ALLOW)}")
    while True:
        try:
            client, _ = srv.accept()
        except OSError:
            continue
        threading.Thread(target=handle, args=(client,), daemon=True).start()


if __name__ == "__main__":
    main()
