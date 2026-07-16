import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()
p = ds.upload("verify_job.py", "Resources", overwrite=True)
print("uploaded:", p, "exists:", ds.exists("Resources/verify_job.py"), flush=True)

job_api = project.get_job_api()
cfg = job_api.get_configuration("PYTHON")
cfg["appPath"] = f"/Projects/{project.name}/Resources/verify_job.py"
job = job_api.create_job("verifyjob72af4e", cfg)
try:
    execution = job.run(await_termination=True)
    print("final status:", execution.final_status, "success:", execution.success, flush=True)
except Exception as e:
    print("job failed:", e, flush=True)
    execution = sorted(job.get_executions(), key=lambda x: x.id)[-1]
out_path, err_path = execution.download_logs()
for pth in (out_path, err_path):
    print("=====", pth, "=====")
    print(open(pth).read()[-6000:])
