import os
# gRPC cannot use the socks5h GRPC_PROXY; point it at the working HTTP proxy.
_httpp = os.environ.get("HTTPS_PROXY", "")
if _httpp:
    os.environ["GRPC_PROXY"] = _httpp
    os.environ["grpc_proxy"] = _httpp
import google.cloud.aiplatform as aiplatform

PROJECT = os.environ['GCP_PROJECT']
LOCATION = os.environ['GCP_LOCATION']
PREFIX = os.environ['MLPAB_GCP_PREFIX']
STAGING = "gs://cloud-ai-platform-5dcfee9a-d8bf-457a-8b19-6d8f5db58035"

aiplatform.init(project=PROJECT, location=LOCATION, staging_bucket=STAGING,
                api_transport="rest")

with open("data/failing_job.py") as f:
    script_src = f.read()

# Materialize the exact provided script inside the container and run it.
bash_cmd = (
    "cat > /tmp/failing_job.py << 'MLPAB_EOF'\n"
    + script_src
    + "\nMLPAB_EOF\n"
    + "python /tmp/failing_job.py"
)

display_name = f"{PREFIX}_flakycdcd16"
container_uri = "europe-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest"

worker_pool_specs = [
    {
        "machine_spec": {"machine_type": "n1-standard-4"},
        "replica_count": 1,
        "container_spec": {
            "image_uri": container_uri,
            "command": ["bash", "-c", bash_cmd],
            "args": [],
        },
    }
]

job = aiplatform.CustomJob(
    display_name=display_name,
    worker_pool_specs=worker_pool_specs,
    labels={"job": "flakycdcd16"},
)

print("Launching CustomJob (expected to FAIL)...", flush=True)
try:
    job.run(sync=True)
except Exception as e:
    print("job.run raised (expected):", type(e).__name__, str(e)[:300], flush=True)

print("RESOURCE_NAME:", job.resource_name, flush=True)
print("DISPLAY_NAME:", job.display_name, flush=True)
print("STATE:", job.state, flush=True)
