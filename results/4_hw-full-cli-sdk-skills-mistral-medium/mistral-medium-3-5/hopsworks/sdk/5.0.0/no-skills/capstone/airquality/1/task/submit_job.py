#!/usr/bin/env python3
"""
Submit the training job to Hopsworks platform.
"""
import hopsworks

print("Connecting to Hopsworks...")
hopsworks.login()

# Get job API
job_api = hopsworks.project.job_api.JobApi()

# Get Python configuration
config = job_api.get_configuration("PYTHON")

# Configure the job
config["appPath"] = "Resources/train_airq_v3.py"
config["name"] = "airq_training_job"

# Set resource limits
config["resourceConfig"]["cores"] = 2
config["resourceConfig"]["memory"] = 4096

print("Creating job...")
job_name = "airq_training_job_v2_2ce555"
job = job_api.create_job(job_name, config)

print(f"Job created: {job.name}")
print(f"Job ID: {job.id}")

print("Launching job...")
job = job_api.get(job_name)
job_api.launch(job_name)

print("Job launched successfully")
print("Waiting for job to complete...")

# Wait for job to complete
import time
while True:
    executions = job_api.last_execution(job)
    if len(executions) > 0:
        status = executions[0]
        print(f"Job status: {status.state}")
        if status.state in ["FINISHED", "FAILED", "KILLED"]:
            break
    time.sleep(10)

print(f"Final job status: {status.state}")
if status.state == "FINISHED":
    print("Job completed successfully!")
else:
    print(f"Job failed with state: {status.state}")
    # Get logs
    print("\nJob logs:")
    if hasattr(status, 'log') and status.log:
        print(status.log)
