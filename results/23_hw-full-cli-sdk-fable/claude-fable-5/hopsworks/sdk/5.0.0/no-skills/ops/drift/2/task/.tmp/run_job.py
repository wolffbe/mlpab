import hw_env  # noqa: F401
import hopsworks

proj = hopsworks.login()
ds = proj.get_dataset_api()
ja = proj.get_job_api()

try:
    ds.mkdir("Resources/drift_task")
except Exception:
    pass
ds.upload("data/features.csv", "Resources/drift_task", overwrite=True)
ds.upload(".tmp/drift_job.py", "Resources/drift_task", overwrite=True)
print("uploaded inputs")

cfg = ja.get_configuration("PYTHON")
cfg["appPath"] = "/Projects/%s/Resources/drift_task/drift_job.py" % proj.name
cfg["resourceConfig"]["memory"] = 4096
cfg["resourceConfig"]["cores"] = 2.0

job = ja.create_job("drift_investigation", cfg)
execution = job.run(await_termination=True)
print("state:", execution.state, "final:", execution.final_status)
try:
    out, err = execution.get_logs()
    print("=== STDOUT (tail) ===")
    print(out[-8000:] if out else out)
    print("=== STDERR (tail) ===")
    print(err[-3000:] if err else err)
except Exception as e:
    print("log fetch failed:", e)
