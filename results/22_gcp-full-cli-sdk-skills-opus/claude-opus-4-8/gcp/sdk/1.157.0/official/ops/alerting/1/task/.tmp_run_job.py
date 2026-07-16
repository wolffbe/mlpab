import os
import google.cloud.aiplatform as aiplatform

PROJECT = os.environ['GCP_PROJECT']
LOCATION = os.environ['GCP_LOCATION']
PREFIX = os.environ['MLPAB_GCP_PREFIX']
STAGING = "gs://cloud-ai-platform-5dcfee9a-d8bf-457a-8b19-6d8f5db58035"

aiplatform.init(project=PROJECT, location=LOCATION, staging_bucket=STAGING)

display_name = f"{PREFIX}_flakycdcd16"
container_uri = "europe-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest"

job = aiplatform.CustomJob.from_local_script(
    display_name=display_name,
    script_path="data/failing_job.py",
    container_uri=container_uri,
    replica_count=1,
    machine_type="n1-standard-4",
    labels={"job": "flakycdcd16"},
)

print("Created job object, launching run (expected to FAIL)...", flush=True)
try:
    job.run(sync=True)
except Exception as e:
    print("job.run raised (expected):", type(e).__name__, str(e)[:300], flush=True)

print("RESOURCE_NAME:", job.resource_name, flush=True)
print("DISPLAY_NAME:", job.display_name, flush=True)
print("STATE:", job.state, flush=True)
