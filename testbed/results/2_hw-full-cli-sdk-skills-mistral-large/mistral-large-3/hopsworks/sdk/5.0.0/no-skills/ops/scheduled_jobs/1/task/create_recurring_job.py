"""
Create a recurring job named `heartbeatd45c2d` on Hopsworks that runs `data/heartbeat.py` periodically.
"""
import hopsworks
import os
import json

# Connect to Hopsworks
project = hopsworks.login()
jobs = project.get_jobs_api()
dataset_api = project.get_dataset_api()

# Upload the script to the project
script_path = "data/heartbeat.py"
upload_path = "Resources/heartbeat.py"
dataset_api.upload(script_path, "Resources", overwrite=True)
print(f"Uploaded {script_path} to {upload_path}")

# Define the job
job_name = "heartbeatd45c2d"

# Get a valid Python job configuration template
job_config = jobs.get_configuration("PYTHON")
job_config["name"] = job_name
job_config["enabled"] = True
job_config["schedule"] = {
    "cron_expression": "0 * * * *",  # Hourly
    "enabled": True,
}
job_config["appPath"] = upload_path

# Create or update the job
try:
    job = jobs.create_job(name=job_name, config=job_config)
    print(f"Job '{job_name}' created successfully.")
except Exception as e:
    print(f"Error creating job: {e}")
    raise

# Trigger a run to ensure it works
try:
    job = jobs.get_job(job_name)
    execution = job.run()
    print(f"Triggered run for job '{job_name}'. Execution ID: {execution.id}")
    print(f"Run completed with state: {execution.state}")
except Exception as e:
    print(f"Error running job: {e}")
    raise

# Write the deliverable
with open("submission/answers.json", "w") as f:
    json.dump({"job_name": job_name}, f)
    print("Deliverable written to submission/answers.json.")