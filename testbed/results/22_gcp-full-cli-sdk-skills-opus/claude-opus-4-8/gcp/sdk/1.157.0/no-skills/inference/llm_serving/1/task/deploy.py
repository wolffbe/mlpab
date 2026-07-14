import os, json, time
import google.cloud.aiplatform as aiplatform
from google.cloud import storage

STAGING_BUCKET = "cloud-ai-platform-5dcfee9a-d8bf-457a-8b19-6d8f5db58035"  # ***REDACTED***

PROJECT  = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
PREFIX   = os.environ["MLPAB_GCP_PREFIX"]
ENDPOINT_NAME = "scorerd51052"

ENDPOINT_DISPLAY = f"{PREFIX}_{ENDPOINT_NAME}"
MODEL_DISPLAY = f"{PREFIX}_scorerd51052_model"

# Prebuilt Vertex sklearn prediction container. It loads model.pkl via
# pickle.load and serves obj.predict(np.asarray(instances)); postprocess()
# returns {"predictions": result.tolist()}. We ship a self-contained pickled
# predictor whose .predict reproduces data/scorer.py exactly (same constants,
# same math.sin trigram sum, same round(...,6)).
IMAGE = "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-6:latest"

# Expression evaluated at load time -> an object with a .predict method. Uses
# only builtins + numpy/math (present in the container). No custom module needed.
EXPR = (
    "type('S',(object,),{'predict': (lambda self, X: __import__('numpy').array("
    "[round(sum(__import__('math').sin("
    "1.160441*ord(t[i])+0.853525*ord(t[i+1])+1.347386*ord(t[i+2])+0.351469"
    ") for i in range(len(t)-2)),6) for t0 in X for t in (str(t0),)]))})()"
)


def build_pickle(expr):
    """Hand-build a pickle (proto 2) that unpickles to eval(expr)."""
    def le4(n):
        return bytes([n & 0xFF, (n >> 8) & 0xFF, (n >> 16) & 0xFF, (n >> 24) & 0xFF])
    eb = expr.encode("utf-8")
    return (b"\x80\x02" + b"c" + b"builtins\n" + b"eval\n"
            + b"X" + le4(len(eb)) + eb + b"\x85" + b"R" + b".")


def cleanup():
    try:
        for ep in aiplatform.Endpoint.list(filter=f'display_name="{ENDPOINT_DISPLAY}"'):
            print("Cleaning old endpoint", ep.resource_name, flush=True)
            try:
                ep.undeploy_all(sync=True)
            except Exception as e:
                print("  undeploy_all:", e, flush=True)
            ep.delete(force=True, sync=True)
    except Exception as e:
        print("endpoint cleanup:", e, flush=True)
    try:
        for m in aiplatform.Model.list(filter=f'display_name="{MODEL_DISPLAY}"'):
            print("Cleaning old model", m.resource_name, flush=True)
            m.delete(sync=True)
    except Exception as e:
        print("model cleanup:", e, flush=True)


def main():
    aiplatform.init(project=PROJECT, location=LOCATION, api_transport="rest")
    cleanup()

    # Upload the self-contained pickled predictor as model.pkl.
    prefix_path = f"{PREFIX}_scorerd51052"
    sc = storage.Client(project=PROJECT)
    bucket = sc.bucket(STAGING_BUCKET)
    # remove any stale artifacts under the prefix
    for b in sc.list_blobs(STAGING_BUCKET, prefix=prefix_path + "/"):
        b.delete()
    pkl = build_pickle(EXPR)
    bucket.blob(f"{prefix_path}/model.pkl").upload_from_string(pkl)
    artifact_uri = f"gs://{STAGING_BUCKET}/{prefix_path}"
    print("artifact_uri:", artifact_uri, "pickle bytes:", len(pkl), flush=True)

    print("Uploading model ...", flush=True)
    model = aiplatform.Model.upload(
        display_name=MODEL_DISPLAY,
        serving_container_image_uri=IMAGE,
        artifact_uri=artifact_uri,
    )
    print("Model resource:", model.resource_name, flush=True)

    print("Creating endpoint ...", flush=True)
    endpoint = aiplatform.Endpoint.create(display_name=ENDPOINT_DISPLAY)
    print("Endpoint resource:", endpoint.resource_name, flush=True)

    print("Deploying model to endpoint (this can take several minutes) ...", flush=True)
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=f"{PREFIX}_scorerd51052_deployed",
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=1,
        traffic_percentage=100,
        sync=True,
    )
    print("Deployment complete.", flush=True)
    time.sleep(5)

    payloads = json.load(open("data/payloads.json"))
    responses = []
    for p in payloads:
        resp = endpoint.predict(instances=[p])
        pred = resp.predictions[0]
        if isinstance(pred, dict):
            val = pred.get("score")
        else:
            val = pred
        responses.append(val)
        print("payload ->", val, flush=True)

    os.makedirs("submission", exist_ok=True)
    out = {"endpoint_name": ENDPOINT_NAME, "responses": responses}
    with open("submission/answers.json", "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE submission/answers.json:", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
