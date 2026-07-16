"""Hopsworks agent: real-time endpoint serving the deterministic trigram scorer.

Agent scripts are *run directly* by the serving runtime, so this module starts
its own HTTP server on port 8080 (the port KServe's queue-proxy probes) and
speaks the KServe v1 inference protocol:

    POST /v1/models/<name>:predict   body {"instances": [<text>, ...]}
                                     -> {"predictions": [{"score": <float>}, ...]}
    GET  /v1/models/<name>           -> {"name": <name>, "ready": true}
    GET  /                            -> liveness/readiness 200

The scorer is the provided pure-stdlib trigram model, embedded verbatim.
A module-level ASGI ``app`` is also exposed so a ``uvicorn module:app`` style
launch works as well.
"""
import json
import math
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------------------
# Provided scorer (verbatim) — fixed trigram log-likelihood model.
# ---------------------------------------------------------------------------
A = 2.045148
B = 2.613667
C = 1.553705
D = -0.98081


def _trigram_weight(tri):
    o0, o1, o2 = (ord(ch) for ch in tri)
    return math.sin(A * o0 + B * o1 + C * o2 + D)


def score(text):
    """Log-likelihood of `text` under the trigram model."""
    ll = 0.0
    for i in range(len(text) - 2):
        ll += _trigram_weight(text[i:i + 3])
    return {"score": round(ll, 6)}


# ---------------------------------------------------------------------------
# Inference logic shared by every transport.
# ---------------------------------------------------------------------------
def _extract_instances(body):
    """Pull the list of text payloads out of a KServe-style request body."""
    if isinstance(body, dict):
        if "instances" in body:
            return body["instances"]
        if "inputs" in body:
            return body["inputs"]
        return [body]
    if isinstance(body, list):
        return body
    return [body]


def _instance_to_text(inst):
    """Coerce a single KServe instance into the text string to score.

    The client wraps payloads as objects/lists (bare strings are rejected
    client-side), so an instance may be a str, a 1-element list, or a dict
    keyed on one of the common field names.
    """
    if isinstance(inst, str):
        return inst
    if isinstance(inst, list):
        return _instance_to_text(inst[0]) if inst else ""
    if isinstance(inst, dict):
        for key in ("text", "input", "data", "payload", "prompt"):
            if key in inst:
                return _instance_to_text(inst[key])
        # fall back to the single value if there is exactly one
        vals = list(inst.values())
        return _instance_to_text(vals[0]) if vals else ""
    return str(inst)


def run_inference(body):
    instances = _extract_instances(body)
    return {"predictions": [score(_instance_to_text(t)) for t in instances]}


# ---------------------------------------------------------------------------
# Stdlib HTTP server (used when the script is run directly).
# ---------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        # Health / readiness for liveness probes and KServe model-ready checks.
        self._send(200, {"name": _MODEL_NAME, "ready": True})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            body = json.loads(raw.decode("utf-8")) if raw else {}
            self._send(200, run_inference(body))
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": str(exc)})

    def log_message(self, *args):  # quiet the default access log
        pass


_MODEL_NAME = os.environ.get("MODEL_NAME") or os.environ.get(
    "DEPLOYMENT_NAME", "scorerd0462a"
)


def _serve():
    port = int(
        os.environ.get("PORT")
        or os.environ.get("PREDICTOR_PORT")
        or os.environ.get("SERVING_PORT")
        or 8080
    )
    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    print(f"scorer agent listening on 0.0.0.0:{port}", flush=True)
    server.serve_forever()


# ---------------------------------------------------------------------------
# Optional ASGI app (used if the runtime launches `uvicorn module:app`).
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI, Request

    app = FastAPI()

    @app.get("/")
    @app.get("/v1/models/{model}")
    async def _ready(model: str = _MODEL_NAME):
        return {"name": model, "ready": True}

    @app.post("/v1/models/{model}:predict")
    @app.post("/{full_path:path}")
    async def _predict(request: Request, model: str = _MODEL_NAME, full_path: str = ""):
        body = await request.json()
        return run_inference(body)
except Exception:  # noqa: BLE001 — FastAPI not importable; stdlib server still works
    app = None


if __name__ == "__main__":
    _serve()
