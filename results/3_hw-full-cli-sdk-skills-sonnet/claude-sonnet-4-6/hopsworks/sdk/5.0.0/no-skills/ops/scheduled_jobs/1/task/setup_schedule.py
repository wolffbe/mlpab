"""Set schedule and trigger execution of the heartbeat job."""
import hopsworks
import time

project = hopsworks.login()
jobs_api = project.get_jobs_api()

job_name = "heartbeat76bfab"

# Get the existing job
job = jobs_api.get_job(job_name)
print(f"Got job: {job.name}")

# Set up a recurring schedule (hourly) - use camelCase keys as the API expects
print("Setting up hourly schedule...")
schedule_config = {
    "id": None,  # None means POST (create new)
    "enabled": True,
    "startDateTime": int(time.time() * 1000),  # milliseconds
    "cronExpression": "0 0 * ? * * *",  # every hour
    "catchup": False,
    "maxActiveRuns": 1,
}
schedule = jobs_api.create_or_update_schedule_job(job_name, schedule_config)
print(f"Schedule created: {schedule}")

# Trigger one run now
print("Triggering a run now...")
execution = job.run(await_termination=True)
print(f"Execution attrs: {[a for a in dir(execution) if not a.startswith('_')]}")
try:
    print(f"Execution state: {execution.state}")
except:
    pass
print("Done!")
