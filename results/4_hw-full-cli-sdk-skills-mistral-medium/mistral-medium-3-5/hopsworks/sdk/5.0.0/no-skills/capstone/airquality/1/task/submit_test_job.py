#!/usr/bin/env python3
import hopsworks

print("Connecting to Hopsworks...")
hopsworks.login()

job_api = hopsworks.project.job_api.JobApi()
config = job_api.get_configuration("PYTHON")
config["appPath"] = "Resources/test_job.py"
config["name"] = "test_job"
config["resourceConfig"]["cores"] = 1
config["resourceConfig"]["memory"] = 2048

print("Creating job...")
job_name = "test_job_2ce555"
job = job_api.create_job(job_name, config)

print(f"Job created: {job.name}")
print("Launching job...")
job = job_api.get(job_name)
job_api.launch(job_name)

print("Waiting for job to complete...")
import time
while True:
    executions = job_api.last_execution(job)
    if len(executions) > 0:
        status = executions[0]
        print(f"Job status: {status.state}")
        if status.state in ["FINISHED", "FAILED", "KILLED"]:
            break
    time.sleep(5)

print(f"Final job status: {status.state}")
