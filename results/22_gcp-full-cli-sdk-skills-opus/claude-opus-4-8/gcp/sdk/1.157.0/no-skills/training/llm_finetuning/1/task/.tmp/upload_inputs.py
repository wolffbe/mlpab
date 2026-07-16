import os
from google.cloud import storage

PROJECT = os.environ['GCP_PROJECT']
PREFIX = os.environ['MLPAB_GCP_PREFIX']
BUCKET = "cloud-ai-platform-5dcfee9a-d8bf-457a-8b19-6d8f5db58035"

client = storage.Client(project=PROJECT)
bucket = client.bucket(BUCKET)
base = f"{PREFIX}/ftjob2e5343"
in_prefix = f"{base}/input"

for local, name in [
    ("data/base_model.npz", "base_model.npz"),
    ("data/finetune.txt", "finetune.txt"),
    ("data/eval.txt", "eval.txt"),
    ("data/finetune_model.py", "finetune_model.py"),
]:
    blob = bucket.blob(f"{in_prefix}/{name}")
    blob.upload_from_filename(local)
    print("uploaded", blob.name)

print("INPUT_URI", f"gs://{BUCKET}/{in_prefix}")
print("OUTPUT_URI", f"gs://{BUCKET}/{base}/output")
