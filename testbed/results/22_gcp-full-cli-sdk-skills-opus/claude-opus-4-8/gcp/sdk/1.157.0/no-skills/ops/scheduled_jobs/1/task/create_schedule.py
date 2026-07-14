import os
import google.cloud.aiplatform as aiplatform

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]

BASE_NAME = "heartbeat518d73"
DISPLAY = f"{PREFIX}_{BASE_NAME}"          # prefixed, still contains heartbeat518d73
PIPELINE_ROOT = "gs://cloud-ai-platform-5dcfee9a-d8bf-457a-8b19-6d8f5db58035/pipeline_root/heartbeat"
TEMPLATE = ".tmp/heartbeat_pipeline.json"

SA = "mlpab-sa@***REDACTED***.iam.gserviceaccount.com"

aiplatform.init(project=PROJECT, location=LOCATION, staging_bucket=PIPELINE_ROOT,
                api_transport="rest")

# Remove any prior schedule with this display name (idempotent re-create).
for s in aiplatform.PipelineJobSchedule.list():
    if s.display_name == DISPLAY:
        print("Deleting prior schedule:", s.resource_name)
        s.delete()

# --- Create the recurring (scheduled) job ---
pj = aiplatform.PipelineJob(
    display_name=DISPLAY,
    template_path=TEMPLATE,
    pipeline_root=PIPELINE_ROOT,
    enable_caching=False,
)

schedule = pj.create_schedule(
    display_name=DISPLAY,
    cron="0 * * * *",          # hourly -> recurring
    max_concurrent_run_count=1,
    max_run_count=None,
    service_account=SA,
)
print("SCHEDULE_RESOURCE:", schedule.resource_name)
print("SCHEDULE_STATE:", schedule.state)
print("SCHEDULE_DISPLAY:", schedule.display_name)
