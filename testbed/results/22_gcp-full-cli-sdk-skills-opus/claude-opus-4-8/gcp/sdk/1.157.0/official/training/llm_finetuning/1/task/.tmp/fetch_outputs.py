import os, json
from google.cloud import storage

BUCKET = "cloud-ai-platform-5dcfee9a-d8bf-457a-8b19-6d8f5db58035"
PREFIX = os.environ["MLPAB_GCP_PREFIX"]
OUT = f"{PREFIX}_ftjob2e5343/outputs"

client = storage.Client(project=os.environ["GCP_PROJECT"])
bkt = client.bucket(BUCKET)

os.makedirs(".tmp/out", exist_ok=True)
for fn in ["finetuned_model.npz", "metrics.json"]:
    blob = bkt.blob(f"{OUT}/{fn}")
    blob.download_to_filename(f".tmp/out/{fn}")
    print("downloaded", fn, blob.size, "bytes")

metrics = json.load(open(".tmp/out/metrics.json"))
print("METRICS", json.dumps(metrics))
