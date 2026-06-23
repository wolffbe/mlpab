#!/usr/bin/env python3
"""Create a recurring scheduled job on Hopsworks platform."""
import hopsworks
import os
import time

# Login to Hopsworks
hopsworks.login()
fs = hopsworks.Fs()

# Read the heartbeat script
with open("data/heartbeat.py", "r") as f:
    script_content = f.read()

# Create the job
job_api = hopsworks.job()

# Upload the script to the platform
fs.upload("data/heartbeat.py", "Resources/heartbeat.py")

# Create a recurring job with hourly schedule
job = job_api.create(
    name="heartbeata1a535",
    entry_point="heartbeat.py",
    description="Heartbeat job that runs periodically",
    schedule="0 * * * *",  # Every hour
    action="new",
    execution_engine="PYTHON",
    app_path="Resources",
    dependencies_archive=None,
    interpreter="python3",
    args=None,
    env_vars=None,
    gpus=0,
    instances=1,
    memory=512,
    disk=1,
    cpu=1,
    max_runtime=60,
    retries=0,
)

print(f"Job created: {job.name} with ID: {job.id}")

# Wait for at least one run to complete
print("Waiting for first run to complete...")
time.sleep(60)  # Wait for the first scheduled run

# Check job runs
runs = job_api.get_runs(job.id)
print(f"Found {len(runs)} runs for job {job.name}")

# Wait a bit more if no runs yet
if len(runs) == 0:
    print("No runs yet, waiting more...")
    time.sleep(60)
    runs = job_api.get_runs(job.id)
    print(f"Found {len(runs)} runs for job {job.name}")

# Create submission directory and write answers
os.makedirs("submission", exist_ok=True)
import json
with open("submission/answers.json", "w") as f:
    json.dump({"job_name": "heartbeata1a535"}, f)

print("Done! submission/answers.json written.")
