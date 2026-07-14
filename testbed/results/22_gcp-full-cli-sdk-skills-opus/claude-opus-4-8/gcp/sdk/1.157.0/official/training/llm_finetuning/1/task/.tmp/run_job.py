import os
import google.cloud.aiplatform as aiplatform

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]
BUCKET = "cloud-ai-platform-5dcfee9a-d8bf-457a-8b19-6d8f5db58035"

BASE = f"gs://{BUCKET}/{PREFIX}_ftjob2e5343"
INPUT_URI = f"{BASE}/inputs"
OUTPUT_URI = f"{BASE}/outputs"

aiplatform.init(project=PROJECT, location=LOCATION, staging_bucket=f"gs://{BUCKET}")

CONTAINER = "europe-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest"

job = aiplatform.CustomJob.from_local_script(
    display_name=f"{PREFIX}_ftjob2e5343",
    script_path=".tmp/train_wrapper.py",
    container_uri=CONTAINER,
    requirements=["google-cloud-storage", "numpy"],
    environment_variables={"INPUT_URI": INPUT_URI, "OUTPUT_URI": OUTPUT_URI},
    machine_type="n1-standard-4",
    replica_count=1,
    base_output_dir=f"{BASE}/job_output",
    staging_bucket=f"gs://{BUCKET}",
)

job.run(sync=True)
print("JOB_STATE", job.state)
print("JOB_RESOURCE", job.resource_name)
print("JOB_DISPLAY", job.display_name)
print("OUTPUT_URI", OUTPUT_URI)
