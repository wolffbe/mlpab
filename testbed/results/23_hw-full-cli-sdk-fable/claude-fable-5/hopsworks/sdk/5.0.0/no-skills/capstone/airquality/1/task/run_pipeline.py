"""Driver: upload data + job script, launch the pipeline job on Hopsworks, stream result."""

import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()

base = "Resources/airq754fa9"
if not ds.exists(base):
    ds.mkdir(base)
ds.upload("data/airquality_history.csv", base, overwrite=True)
ds.upload("data/forecast_days.csv", base, overwrite=True)
ds.upload("pipeline_job.py", base, overwrite=True)
print("uploads done", flush=True)

job_api = project.get_job_api()
cfg = job_api.get_configuration("PYTHON")
cfg["appPath"] = f"/Projects/{project.name}/{base}/pipeline_job.py"
cfg["resourceConfig"]["cores"] = 2.0
cfg["resourceConfig"]["memory"] = 4096
job = job_api.create_job("airq_pipeline_754fa9", cfg)
print("job created:", job.name, flush=True)

execution = job.run(await_termination=True)
print("final status:", execution.final_status, "success:", execution.success, flush=True)

try:
    out_path, err_path = execution.download_logs()
    for label, p in [("STDOUT", out_path), ("STDERR", err_path)]:
        print(f"===== {label} ({p}) =====", flush=True)
        with open(p) as f:
            print(f.read(), flush=True)
except Exception as e:  # noqa: BLE001
    print("log download failed:", e, flush=True)
