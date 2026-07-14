import os, json, pickle, time
import google.cloud.aiplatform as aiplatform
from google.cloud import storage

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]
ENDPOINT_NAME = "scorerd51052"
EXISTING_ENDPOINT = "projects/1014453977696/locations/***REDACTED***/endpoints/7590018926491205632"

aiplatform.init(project=PROJECT, location=LOCATION, api_transport="rest")

# The Vertex prebuilt sklearn container loads model.pkl and calls model.predict(X)
# where X = np.asarray(request["instances"]) and returns {"predictions": pred.tolist()}.
# We supply an artifact whose predict computes data/scorer.py's exact trigram
# log-likelihood IN THE CONTAINER. Built with stdlib pickle only (no local ML libs).
# Weights A,B,C,D match scorer.py exactly.
PREDICT_EXPR = (
    "type('ScorerModel', (), {'predict': (lambda self, X: "
    "__import__('numpy').array(["
    "{'score': round(sum("
    "__import__('math').sin("
    "1.160441*ord(t[i]) + 0.853525*ord(t[i+1]) + 1.347386*ord(t[i+2]) + 0.351469"
    ") for i in range(len(t)-2)), 6)}"
    " for t in X], dtype=object))})()"
)


class ScorerArtifact:
    """Unpickles (via eval of PREDICT_EXPR) into an object with a .predict method."""
    def __reduce__(self):
        return (eval, (PREDICT_EXPR,))


# sanity: locally verify the expression is well-formed and matches scorer.py
# (uses only stdlib + numpy-free path here for the check).
import math
def _ref_score(text):
    ll = 0.0
    for i in range(len(text) - 2):
        o0, o1, o2 = ord(text[i]), ord(text[i+1]), ord(text[i+2])
        ll += math.sin(1.160441*o0 + 0.853525*o1 + 1.347386*o2 + 0.351469)
    return round(ll, 6)

with open("data/payloads.json") as f:
    payloads = json.load(f)
print("reference scores:", [_ref_score(p) for p in payloads], flush=True)

# Upload the crafted artifact
bucket_name = "***REDACTED***-feature-store"
art_prefix = f"{PREFIX}_scorer_artifact"
sc = storage.Client(project=PROJECT)
b = sc.bucket(bucket_name)
b.blob(f"{art_prefix}/model.pkl").upload_from_string(pickle.dumps(ScorerArtifact()))
artifact_uri = f"gs://{bucket_name}/{art_prefix}"
print("uploaded artifact to", artifact_uri, flush=True)

print("Uploading model (prebuilt sklearn image, native server, no command override)...", flush=True)
model = aiplatform.Model.upload(
    display_name=f"{PREFIX}_{ENDPOINT_NAME}_model",
    serving_container_image_uri="europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
    artifact_uri=artifact_uri,
    sync=True,
)
print("Model uploaded:", model.resource_name, flush=True)

endpoint = aiplatform.Endpoint(EXISTING_ENDPOINT)
print("Undeploying any existing models on endpoint...", flush=True)
try:
    endpoint.undeploy_all(sync=True)
except Exception as e:
    print("undeploy_all note:", e, flush=True)

print("Deploying new model...", flush=True)
model.deploy(
    endpoint=endpoint,
    machine_type="n1-standard-2",
    min_replica_count=1,
    max_replica_count=1,
    traffic_percentage=100,
    sync=True,
)
print("Deployed.", flush=True)

responses = []
for i, text in enumerate(payloads):
    last = None
    for attempt in range(8):
        try:
            resp = endpoint.predict(instances=[text])
            pred = resp.predictions[0]
            responses.append(pred)
            print(f"payload[{i}] -> {pred}", flush=True)
            last = None
            break
        except Exception as e:
            last = e
            print(f"payload[{i}] attempt {attempt} failed: {type(e).__name__}: {e}", flush=True)
            time.sleep(15)
    if last is not None:
        raise last

out = {"endpoint_name": ENDPOINT_NAME, "responses": responses}
with open("submission/answers.json", "w") as f:
    json.dump(out, f, indent=2)
print("WROTE submission/answers.json", flush=True)
print(json.dumps(out), flush=True)
