import os
import google.cloud.aiplatform as aiplatform

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]

BUCKET = "gs://cloud-ai-platform-5dcfee9a-d8bf-457a-8b19-6d8f5db58035"
PIPELINE_ROOT = f"{BUCKET}/heartbeat518d73"
TEMPLATE = os.path.join(os.path.dirname(__file__), "heartbeat_pipeline.json")

DISPLAY = f"{PREFIX}_heartbeat518d73"

aiplatform.init(project=PROJECT, location=LOCATION, staging_bucket=BUCKET,
                api_transport="rest")

# Build a PipelineJob from the hand-authored template, then attach a recurring
# (cron) schedule to make it a RECURRING scheduled job.
job = aiplatform.PipelineJob(
    display_name=DISPLAY,
    template_path=TEMPLATE,
    pipeline_root=PIPELINE_ROOT,
    enable_caching=False,
)

schedule = job.create_schedule(
    cron="0 * * * *",            # hourly
    display_name=DISPLAY,
    max_concurrent_run_count=1,
    allow_queueing=False,
)

print("SCHEDULE_RESOURCE_NAME", schedule.resource_name)
print("SCHEDULE_DISPLAY_NAME", schedule.display_name)
print("SCHEDULE_STATE", schedule.state)
print("SCHEDULE_CRON", schedule.cron)
