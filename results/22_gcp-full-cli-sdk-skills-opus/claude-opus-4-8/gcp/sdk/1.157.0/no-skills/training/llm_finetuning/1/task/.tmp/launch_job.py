import os
import google.cloud.aiplatform as aiplatform

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]
BUCKET = "cloud-ai-platform-5dcfee9a-d8bf-457a-8b19-6d8f5db58035"
base = f"gs://{BUCKET}/{PREFIX}/ftjob2e5343"
IN = f"{base}/input"
OUT = f"{base}/output"
WRAPPER_URI = f"{IN}/wrapper_job.py"

aiplatform.init(project=PROJECT, location=LOCATION, staging_bucket=f"gs://{BUCKET}")

bootstrap = (
    "import os,runpy\n"
    "from google.cloud import storage\n"
    "u=os.environ['WRAPPER_URI']; bkt,_,pfx=u[5:].partition('/')\n"
    "storage.Client().bucket(bkt).blob(pfx).download_to_filename('wrapper_job.py')\n"
    "runpy.run_path('wrapper_job.py', run_name='__main__')\n"
)

worker_pool_specs = [{
    "machine_spec": {"machine_type": "n1-standard-4"},
    "replica_count": 1,
    "container_spec": {
        "image_uri": "europe-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest",
        "command": ["python", "-c", bootstrap],
        "args": [],
        "env": [
            {"name": "WRAPPER_URI", "value": WRAPPER_URI},
            {"name": "JOB_INPUT_URI", "value": IN},
            {"name": "JOB_OUTPUT_URI", "value": OUT},
        ],
    },
}]

job = aiplatform.CustomJob(
    display_name=f"{PREFIX}_ftjob2e5343",
    worker_pool_specs=worker_pool_specs,
    base_output_dir=f"{base}/aip",
)
job.run(sync=True)
print("STATE", job.state)
print("RESOURCE", job.resource_name)
print("DISPLAY", job.display_name)
