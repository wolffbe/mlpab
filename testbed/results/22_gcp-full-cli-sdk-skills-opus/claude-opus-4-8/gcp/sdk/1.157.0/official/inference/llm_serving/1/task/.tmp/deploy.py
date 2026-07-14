import os, json, sys
import google.cloud.aiplatform as aiplatform

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]
ENDPOINT_NAME = "scorerd51052"

aiplatform.init(project=PROJECT, location=LOCATION, api_transport="rest")

# ---- Inline model server (runs INSIDE the Vertex prediction container) ----
# Pure stdlib re-implementation of data/scorer.py; the scoring executes on the
# platform, not locally. Implements the Vertex online-prediction contract.
SERVER_CODE = r'''
import os, json, math
from http.server import BaseHTTPRequestHandler, HTTPServer

A = 1.160441
B = 0.853525
C = 1.347386
D = 0.351469

def _trigram_weight(tri):
    o0, o1, o2 = (ord(ch) for ch in tri)
    return math.sin(A * o0 + B * o1 + C * o2 + D)

def score(text):
    ll = 0.0
    for i in range(len(text) - 2):
        ll += _trigram_weight(text[i:i + 3])
    return {"score": round(ll, 6)}

PORT = int(os.environ.get("AIP_HTTP_PORT", "8080"))
HEALTH = os.environ.get("AIP_HEALTH_ROUTE", "/health")
PREDICT = os.environ.get("AIP_PREDICT_ROUTE", "/predict")

def _extract_text(inst):
    if isinstance(inst, str):
        return inst
    if isinstance(inst, dict):
        for k in ("text", "content", "input", "payload", "data"):
            if k in inst and isinstance(inst[k], str):
                return inst[k]
        return json.dumps(inst)
    return str(inst)

class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        self._send(200, {"status": "ok"})
    def do_POST(self):
        if self.path == PREDICT or self.path == "/predict":
            n = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(n) if n else b"{}"
            try:
                req = json.loads(raw.decode("utf-8") or "{}")
            except Exception:
                req = {}
            instances = req.get("instances", [])
            preds = [score(_extract_text(it)) for it in instances]
            self._send(200, {"predictions": preds})
        else:
            self._send(404, {"error": "not found"})
    def log_message(self, *a):
        return

if __name__ == "__main__":
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
'''

print("Uploading model with custom serving container...", flush=True)
model = aiplatform.Model.upload(
    display_name=f"{PREFIX}_{ENDPOINT_NAME}_model",
    serving_container_image_uri="gcr.io/deeplearning-platform-release/base-cpu:latest",
    serving_container_command=["python", "-c", SERVER_CODE],
    serving_container_ports=[8080],
    serving_container_predict_route="/predict",
    serving_container_health_route="/health",
    sync=True,
)
print("Model uploaded:", model.resource_name, flush=True)

print("Creating endpoint...", flush=True)
endpoint = aiplatform.Endpoint.create(display_name=f"{PREFIX}_{ENDPOINT_NAME}")
print("Endpoint created:", endpoint.resource_name, flush=True)

print("Deploying model to endpoint (this is the slow step)...", flush=True)
model.deploy(
    endpoint=endpoint,
    machine_type="n1-standard-2",
    min_replica_count=1,
    max_replica_count=1,
    traffic_percentage=100,
    sync=True,
)
print("Deployed. Endpoint:", endpoint.resource_name, flush=True)

with open("data/payloads.json") as f:
    payloads = json.load(f)

responses = []
for i, text in enumerate(payloads):
    resp = endpoint.predict(instances=[text])
    pred = resp.predictions[0]
    print(f"payload[{i}] -> {pred}", flush=True)
    responses.append(pred)

out = {"endpoint_name": ENDPOINT_NAME, "responses": responses}
with open("submission/answers.json", "w") as f:
    json.dump(out, f, indent=2)
print("WROTE submission/answers.json", flush=True)
print(json.dumps(out), flush=True)
