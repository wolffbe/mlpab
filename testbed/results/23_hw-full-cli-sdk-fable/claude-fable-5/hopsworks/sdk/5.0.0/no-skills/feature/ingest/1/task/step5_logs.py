import hopsworks

project = hopsworks.login()
jobs_api = project.get_job_api()
job = jobs_api.get_job("ingest_transactions82e347")
execs = job.get_executions()
ex = sorted(execs, key=lambda e: e.id)[-1]
print("execution:", ex.id, ex.state, ex.final_status)
print("methods:", [m for m in dir(ex) if not m.startswith("_")])
try:
    ex.download_logs()
except Exception as e:
    print("download_logs failed:", e)
print("stdout_path:", getattr(ex, "stdout_path", None))
print("stderr_path:", getattr(ex, "stderr_path", None))
ds = project.get_dataset_api()
for p in (getattr(ex, "stdout_path", None), getattr(ex, "stderr_path", None)):
    if p:
        try:
            print("=== ", p, " ===")
            print(ds.read_content(p).decode() if hasattr(ds, "read_content") else "")
        except Exception as e:
            print("read failed:", e)
