import os
import google.cloud.aiplatform as aiplatform
from google.cloud.aiplatform import pipeline_job_schedules

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]

BUCKET = "gs://cloud-ai-platform-5dcfee9a-d8bf-457a-8b19-6d8f5db58035"
PIPELINE_ROOT = f"{BUCKET}/heartbeat518d73"
TEMPLATE = os.path.join(os.path.dirname(__file__), "heartbeat_pipeline.json")
DISPLAY = f"{PREFIX}_heartbeat518d73"
SA = "mlpab-sa@***REDACTED***.iam.gserviceaccount.com"

aiplatform.init(project=PROJECT, location=LOCATION, staging_bucket=BUCKET,
                api_transport="rest")

# Delete any prior heartbeat schedule for this prefix (created without an SA).
for s in pipeline_job_schedules.PipelineJobSchedule.list():
    if s.display_name == DISPLAY:
        print("DELETING stale schedule", s.resource_name)
        s.delete()

# Recreate the recurring schedule, this time with a runnable service account.
job = aiplatform.PipelineJob(
    display_name=DISPLAY, template_path=TEMPLATE,
    pipeline_root=PIPELINE_ROOT, enable_caching=False,
)
schedule = job.create_schedule(
    cron="0 * * * *", display_name=DISPLAY,
    max_concurrent_run_count=1, allow_queueing=False,
    service_account=SA,
)
print("SCHEDULE_RESOURCE_NAME", schedule.resource_name)
print("SCHEDULE_DISPLAY_NAME", schedule.display_name)
print("SCHEDULE_STATE", schedule.state)
print("SCHEDULE_CRON", schedule.cron)

# Submit one immediate run so a run completes within the budget.
run = aiplatform.PipelineJob(
    display_name=DISPLAY, template_path=TEMPLATE,
    pipeline_root=PIPELINE_ROOT, enable_caching=False,
)
run.submit(service_account=SA)
print("RUN_SUBMITTED", run.resource_name)
run.wait()
print("RUN_FINAL_STATE", run.state)
