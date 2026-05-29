"""Remote HTTP MCP server (official mcp SDK / FastMCP, streamable-http) exposing
mlkit's fake AutoML as tools: `fit` and `predict`.

Subcommands: `serve` (run HTTP server), `ensure` (start in background if down),
`login` (fake auth), `--version`. `--version`/`ensure`/`login` work without the
`mcp` package installed (only `serve` imports it).
"""
from __future__ import annotations
import argparse
import os
import socket
import subprocess
import sys
import time
from urllib.parse import urlparse


def mcp_url():
    return os.environ.get("MLKIT_MCP_URL", "http://127.0.0.1:8766/mcp")


def _host_port():
    u = urlparse(mcp_url())
    return (u.hostname or "127.0.0.1", u.port or 8766)


def _is_up():
    host, port = _host_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _ensure(wait=15.0):
    if _is_up():
        return 0
    subprocess.Popen(
        [sys.executable, "-m", "mlkit_mcp.server", "serve"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL, start_new_session=True, env=dict(os.environ),
    )
    end = time.time() + wait
    while time.time() < end:
        if _is_up():
            return 0
        time.sleep(0.3)
    print("mlkit MCP server did not come up at " + mcp_url(), file=sys.stderr)
    return 1


def _serve():
    from mcp.server.fastmcp import FastMCP
    from mlkit_mcp import _client

    host, port = _host_port()
    mcp = FastMCP("mlkit", host=host, port=port)

    @mcp.tool()
    def fit(data_dir: str = "data") -> str:
        """Train an mlkit model on <data_dir>/train.csv (stub — not implemented)."""
        return _client.fit(data_dir)

    @mcp.tool()
    def predict(
        model_id: str = "",
        data_dir: str = "data",
        out_path: str = "submission/submission.csv",
    ) -> str:
        """Write submission.csv for a fitted model_id (stub — not implemented)."""
        return _client.predict(model_id, data_dir, out_path)

    mcp.run(transport="streamable-http")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="mlkit-mcp", description="mlkit MCP server")
    p.add_argument("--version", action="version", version="mlkit-mcp 0.1.0")
    p.add_argument("cmd", nargs="?", choices=["serve", "ensure", "login"], default="ensure")
    a = p.parse_args(argv)
    if a.cmd == "login":
        from mlkit_mcp import _client
        _client.login()
        print("[mlkit-mcp] logged in")
        return 0
    return _serve() if a.cmd == "serve" else _ensure()


if __name__ == "__main__":
    raise SystemExit(main())
