import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()
p = ds.upload("data/requests.csv", "Resources", overwrite=True)
print("uploaded:", p, "exists:", ds.exists("Resources/requests.csv"), flush=True)

job_api = project.get_job_api()
job = job_api.get_job("scoredjob72af4e")
try:
    execution = job.run(await_termination=True)
    print("final status:", execution.final_status, "success:", execution.success, flush=True)
except Exception as e:
    print("job failed:", e, flush=True)
    execution = sorted(job.get_executions(), key=lambda x: x.id)[-1]
out_path, err_path = execution.download_logs()
for pth in (out_path, err_path):
    print("=====", pth, "=====")
    print(open(pth).read()[-12000:])
