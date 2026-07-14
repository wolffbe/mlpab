import os
import google.cloud.aiplatform as aiplatform

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]

DISPLAY = f"{PREFIX}_heartbeat518d73"
PIPELINE_ROOT = "gs://cloud-ai-platform-5dcfee9a-d8bf-457a-8b19-6d8f5db58035/pipeline_root/heartbeat"
TEMPLATE = ".tmp/heartbeat_pipeline.json"

aiplatform.init(project=PROJECT, location=LOCATION, staging_bucket=PIPELINE_ROOT,
                api_transport="rest")

pj = aiplatform.PipelineJob(
    display_name=DISPLAY,
    template_path=TEMPLATE,
    pipeline_root=PIPELINE_ROOT,
    enable_caching=False,
)
pj.submit(service_account="mlpab-sa@***REDACTED***.iam.gserviceaccount.com")
print("RUN_RESOURCE:", pj.resource_name)
pj.wait()
print("FINAL_STATE:", pj.state)
