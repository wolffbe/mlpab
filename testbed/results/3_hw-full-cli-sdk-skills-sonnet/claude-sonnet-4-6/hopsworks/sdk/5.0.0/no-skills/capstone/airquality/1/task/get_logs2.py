import hopsworks
project = hopsworks.login()
jobs_api = project.get_jobs_api()
job = jobs_api.get_job("airq_train_job")
executions = job.get_executions()
last = executions[0]
print(f"Status: {last.final_status}")
log_files = last.download_logs()
import os
stdout_path, stderr_path = log_files
print("=== STDOUT ===")
with open(stdout_path) as f:
    print(f.read())
print("=== STDERR ===")
with open(stderr_path) as f:
    print(f.read())
