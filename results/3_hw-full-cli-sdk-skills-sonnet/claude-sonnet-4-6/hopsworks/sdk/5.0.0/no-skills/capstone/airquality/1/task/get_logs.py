"""Get logs from the last job execution."""
import hopsworks
import os

project = hopsworks.login()
jobs_api = project.get_jobs_api()

job = jobs_api.get_job("airq_train_job")

executions = job.get_executions()
last = executions[0]
print(f"Last execution final_status: {last.final_status}")
print(f"stdout_path: {last.stdout_path}")
print(f"stderr_path: {last.stderr_path}")

try:
    log_dir = last.download_logs()
    print(f"Logs downloaded to: {log_dir}")
    for root, dirs, files in os.walk(log_dir):
        for f in files:
            fpath = os.path.join(root, f)
            print(f"\n=== {fpath} ===")
            with open(fpath) as fp:
                print(fp.read()[:5000])
except Exception as e:
    print(f"download_logs error: {e}")
