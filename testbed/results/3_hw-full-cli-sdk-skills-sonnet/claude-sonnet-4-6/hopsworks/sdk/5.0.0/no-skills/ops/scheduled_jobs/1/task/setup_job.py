"""Create a recurring scheduled heartbeat job on Hopsworks."""
import hopsworks
import json
import time

project = hopsworks.login()
jobs_api = project.get_jobs_api()
dataset_api = project.get_dataset_api()

# Upload the heartbeat script to the project's Resources directory
print("Uploading heartbeat.py...")
dataset_api.upload("data/heartbeat.py", "Resources", overwrite=True)
print("Uploaded successfully.")

# Get Python job configuration
python_config = jobs_api.get_configuration("PYTHON")
print("Python config:", json.dumps(python_config, indent=2))

# Set the script path
python_config["appPath"] = "Resources/heartbeat.py"

# Create the job
job_name = "heartbeat76bfab"
print(f"Creating job {job_name}...")
job = jobs_api.create_job(job_name, python_config)
print(f"Job created: {job}")

# Set up a recurring schedule (hourly)
print("Setting up hourly schedule...")
schedule_config = {
    "enabled": True,
    "start_time": int(time.time() * 1000),  # milliseconds
    "cron_expression": "0 0 * ? * * *",  # every hour at minute 0
}
schedule = jobs_api.create_or_update_schedule_job(job_name, schedule_config)
print(f"Schedule created: {schedule}")

# Trigger one run now
print("Triggering a run...")
execution = job.run(await_termination=True)
print(f"Execution completed: {execution}")
print(f"Execution state: {execution.state if hasattr(execution, 'state') else 'unknown'}")
