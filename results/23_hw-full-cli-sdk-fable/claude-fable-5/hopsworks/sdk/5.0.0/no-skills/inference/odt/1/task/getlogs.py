import hopsworks

project = hopsworks.login()
job_api = project.get_job_api()
job = job_api.get_job("scoredjob72af4e")
execs = job.get_executions()
ex = sorted(execs, key=lambda e: e.id)[-1]
print("execution", ex.id, ex.state, ex.final_status)
out_path, err_path = ex.download_logs()
for pth in (out_path, err_path):
    print("=====", pth, "=====")
    try:
        print(open(pth).read()[-8000:])
    except Exception as e:
        print("read failed:", e)
