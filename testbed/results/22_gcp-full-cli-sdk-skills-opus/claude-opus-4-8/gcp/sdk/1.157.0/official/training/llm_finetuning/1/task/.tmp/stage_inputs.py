import os
from google.cloud import storage

BUCKET = "cloud-ai-platform-5dcfee9a-d8bf-457a-8b19-6d8f5db58035"
PREFIX = os.environ["MLPAB_GCP_PREFIX"]
IN_PREFIX = f"{PREFIX}_ftjob2e5343/inputs"

client = storage.Client(project=os.environ["GCP_PROJECT"])
bkt = client.bucket(BUCKET)
files = {
    "data/base_model.npz": "base_model.npz",
    "data/finetune.txt": "finetune.txt",
    "data/eval.txt": "eval.txt",
    "data/finetune_model.py": "finetune_model.py",
}
for local, name in files.items():
    blob = bkt.blob(f"{IN_PREFIX}/{name}")
    blob.upload_from_filename(local)
    print("uploaded", f"gs://{BUCKET}/{IN_PREFIX}/{name}")
print("INPUT_URI", f"gs://{BUCKET}/{IN_PREFIX}")
