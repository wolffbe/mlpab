import os, json
import google.cloud.aiplatform as aiplatform
from google.cloud import storage

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]
ENDPOINT_NAME = "scorerd51052"
EXISTING_ENDPOINT = "projects/1014453977696/locations/***REDACTED***/endpoints/7590018926491205632"

aiplatform.init(project=PROJECT, location=LOCATION, api_transport="rest")

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

# Dummy artifact dir to satisfy prebuilt-container validation (ignored by our
# overridden command).
bucket_name = "***REDACTED***-feature-store"
art_prefix = f"{PREFIX}_scorer_artifact"
sc = storage.Client(project=PROJECT)
b = sc.bucket(bucket_name)
import pickle
b.blob(f"{art_prefix}/model.pkl").upload_from_string(pickle.dumps({}))
artifact_uri = f"gs://{bucket_name}/{art_prefix}"
print("artifact_uri:", artifact_uri, flush=True)

print("Uploading model (prebuilt image + command override)...", flush=True)
model = aiplatform.Model.upload(
    display_name=f"{PREFIX}_{ENDPOINT_NAME}_model",
    serving_container_image_uri="europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
    artifact_uri=artifact_uri,
    serving_container_command=["python", "-c", SERVER_CODE],
    serving_container_ports=[8080],
    serving_container_predict_route="/predict",
    serving_container_health_route="/health",
    sync=True,
)
print("Model uploaded:", model.resource_name, flush=True)

endpoint = aiplatform.Endpoint(EXISTING_ENDPOINT)
print("Reusing endpoint:", endpoint.resource_name, flush=True)

print("Deploying model to endpoint...", flush=True)
model.deploy(
    endpoint=endpoint,
    machine_type="n1-standard-2",
    min_replica_count=1,
    max_replica_count=1,
    traffic_percentage=100,
    sync=True,
)
print("Deployed.", flush=True)

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
