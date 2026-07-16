import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()
for f in ["data/requests.csv", "data/profiles.csv", "scored_job.py"]:
    p = ds.upload(f, "Resources", overwrite=True)
    print("uploaded:", p, flush=True)

job_api = project.get_job_api()
cfg = job_api.get_configuration("PYTHON")
cfg["appPath"] = f"/Projects/{project.name}/Resources/scored_job.py"
job = job_api.create_job("scoredjob72af4e", cfg)
print("job created:", job.name, flush=True)
execution = job.run(await_termination=True)
print("final status:", execution.final_status, "success:", execution.success, flush=True)
try:
    out_path, err_path = execution.download_logs()
    print("logs:", out_path, err_path, flush=True)
    for pth in (out_path, err_path):
        try:
            print("=====", pth, "=====")
            print(open(pth).read()[-8000:])
        except Exception as e:
            print("log read failed:", e)
except Exception as e:
    print("log download failed:", e)
