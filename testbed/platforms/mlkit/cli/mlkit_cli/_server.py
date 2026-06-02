"""Tiny local "mlkit" platform (stdlib only) — the fake remote service used by
the auth smoke check. Endpoints:
  GET  /health             -> {"status":"ok"}
  POST /login {api_key}    -> 200 if api_key non-empty else 401 (fake auth)

mlkit does no real ML; `fit`/`predict` are client-side stubs (see _client), so
the platform only needs `/health` + `/login`.
"""
from __future__ import annotations
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


def _host_port():
    u = urlparse(os.environ.get("MLKIT_HOST", "http://127.0.0.1:8765"))
    return (u.hostname or "127.0.0.1", u.port or 8765)


class _H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"status": "ok"})
        self._send(404, {"error": "unknown route"})

    def do_POST(self):
        if self.path == "/login":
            body = self._body()
            if body.get("api_key"):
                return self._send(200, {"token": "ok"})
            return self._send(401, {"error": "missing api_key"})
        self._send(404, {"error": "unknown route"})


def main():
    host, port = _host_port()
    ThreadingHTTPServer((host, port), _H).serve_forever()


if __name__ == "__main__":
    main()
