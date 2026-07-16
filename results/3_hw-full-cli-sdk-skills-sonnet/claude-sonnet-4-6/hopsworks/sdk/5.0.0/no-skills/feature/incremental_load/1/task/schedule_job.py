import hopsworks
from datetime import datetime, timezone

project = hopsworks.login()

print("Connected to Hopsworks")

jobs_api = project.get_jobs_api()
job = jobs_api.get_job("incrementaljob811051")
print(f"Got job: {job.name}")

# Schedule daily at midnight UTC (Quartz cron: seconds minutes hours day month weekday year)
# "0 0 0 * * ? *" = at 00:00:00 every day
job_schedule = job.schedule(
    cron_expression="0 0 0 * * ? *",
    start_time=datetime.now(tz=timezone.utc),
)
print(f"Job scheduled: {job_schedule}")
print(f"Schedule details: {job_schedule}")
