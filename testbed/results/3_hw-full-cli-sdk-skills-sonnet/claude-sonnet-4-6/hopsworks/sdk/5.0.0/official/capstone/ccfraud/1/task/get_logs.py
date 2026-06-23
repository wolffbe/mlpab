#!/usr/bin/env python3
import hopsworks

project = hopsworks.login()
jobs_api = project.get_jobs_api()
job = jobs_api.get_job("ccfraud_pipeline")
executions = job.get_executions()
print(f"Found {len(executions)} executions")
exec = executions[0]
print(f"Execution state: {exec.state}")
print(f"Execution attrs: {[a for a in dir(exec) if not a.startswith('_')]}")

# Try to get logs through the job API
try:
    logs = jobs_api.get_execution_log(job, exec.id, log_type="out")
    print("=== STDOUT ===")
    print(str(logs)[-5000:])
except Exception as e:
    print(f"get_execution_log error: {e}")

# Try job's download logs
try:
    exec.download_logs()
    print("Logs downloaded")
except Exception as e:
    print(f"download_logs error: {e}")

# Also try reading logs via the API
try:
    import requests
    resp = requests.get(
        f"https://***REDACTED***:443/hopsworks-api/api/project/{project.id}/jobs/ccfraud_pipeline/executions/{exec.id}/logs",
        headers={"Authorization": "ApiKey " + os.environ.get("HOPSWORKS_API_KEY", "")},
        verify=False
    )
    print("Direct API:", resp.status_code, resp.text[:2000])
except Exception as e:
    print(f"direct api error: {e}")
