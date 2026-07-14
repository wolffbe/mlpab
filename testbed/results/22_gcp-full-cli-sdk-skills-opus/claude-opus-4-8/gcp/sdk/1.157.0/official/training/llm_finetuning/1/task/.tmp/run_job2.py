import os
import google.cloud.aiplatform as aiplatform

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]
BUCKET = "cloud-ai-platform-5dcfee9a-d8bf-457a-8b19-6d8f5db58035"

BASE = f"gs://{BUCKET}/{PREFIX}_ftjob2e5343"
INPUT_URI = f"{BASE}/inputs"
OUTPUT_URI = f"{BASE}/outputs"

aiplatform.init(project=PROJECT, location=LOCATION, staging_bucket=f"gs://{BUCKET}",
                api_transport="rest")

CONTAINER = "europe-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest"

BOOTSTRAP = r'''
import os, runpy
from google.cloud import storage

def split(uri):
    b, _, p = uri[len("gs://"):].partition("/")
    return b, p

client = storage.Client()
in_bucket, in_prefix = split(os.environ["INPUT_URI"].rstrip("/"))
bkt = client.bucket(in_bucket)
for fn in ["base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"]:
    bkt.blob(in_prefix + "/" + fn).download_to_filename(fn)
    print("downloaded", fn, flush=True)

runpy.run_path("finetune_model.py", run_name="__main__")
print("finetune complete", flush=True)

out_bucket, out_prefix = split(os.environ["OUTPUT_URI"].rstrip("/"))
obkt = client.bucket(out_bucket)
for fn in ["finetuned_model.npz", "metrics.json"]:
    obkt.blob(out_prefix + "/" + fn).upload_from_filename(fn)
    print("uploaded", fn, flush=True)
print("DONE", flush=True)
'''

worker_pool_specs = [
    {
        "machine_spec": {"machine_type": "n1-standard-4"},
        "replica_count": 1,
        "container_spec": {
            "image_uri": CONTAINER,
            "command": ["python", "-c", BOOTSTRAP],
            "env": [
                {"name": "INPUT_URI", "value": INPUT_URI},
                {"name": "OUTPUT_URI", "value": OUTPUT_URI},
            ],
        },
    }
]

job = aiplatform.CustomJob(
    display_name=f"{PREFIX}_ftjob2e5343",
    worker_pool_specs=worker_pool_specs,
    staging_bucket=f"gs://{BUCKET}",
    base_output_dir=f"{BASE}/job_output",
)

job.run(sync=True)
print("JOB_STATE", job.state)
print("JOB_RESOURCE", job.resource_name)
print("JOB_DISPLAY", job.display_name)
print("OUTPUT_URI", OUTPUT_URI)
