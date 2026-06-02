"""HTTP client for the local mlkit platform + the (stub) ML ops (stdlib only).

mlkit is a FAKE interface for smoke-testing the testbed plumbing — it does no
real ML. `login` is real (it exercises auth against the local platform), but
`fit` and `predict` are STUBS that return "not implemented" and do nothing.
Their only purpose is to be invokable across cli/sdk/mcp so the harness can
verify the engineer can reach the interface (and count the calls).
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
import urllib.request


def base_url():
    return os.environ.get("MLKIT_HOST", "http://127.0.0.1:8765").rstrip("/")


def _get(path, timeout=5.0):
    with urllib.request.urlopen(base_url() + path, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")


def _post(path, payload, timeout=10.0):
    req = urllib.request.Request(
        base_url() + path, data=json.dumps(payload).encode(),
        method="POST", headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")


def is_up():
    try:
        _get("/health", timeout=1.0)
        return True
    except Exception:
        return False


def ensure_running(wait=10.0):
    if is_up():
        return
    # `__package__._server` so this identical client works in every variant
    # package. stdout/stderr discarded — the platform is a stub for the
    # testbed; failures show up as connection errors from the next request.
    subprocess.Popen(
        [sys.executable, "-m", f"{__package__}._server"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL, start_new_session=True, env=dict(os.environ),
    )
    end = time.time() + wait
    while time.time() < end:
        if is_up():
            return
        time.sleep(0.2)
    raise RuntimeError(f"mlkit platform did not come up at {base_url()}")


def login():
    """Fake auth: send MLKIT_API_KEY to the platform; raises on 401."""
    ensure_running()
    return _post("/login", {"api_key": os.environ.get("MLKIT_API_KEY", "")})


def fit(data_dir="data"):
    """Stub — mlkit does no real ML. Returns a 'not implemented' message."""
    return "mlkit.fit: not implemented (mlkit is a stub smoke-test interface)"


def predict(model_id=None, data_dir="data", out_path="submission/submission.csv"):
    """Stub — does nothing and writes no submission. Returns 'not implemented'."""
    return "mlkit.predict: not implemented (mlkit is a stub smoke-test interface)"
