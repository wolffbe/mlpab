import hopsworks

proj = hopsworks.login()
job_api = proj.get_job_api()
job = job_api.get_job("prediction_monitoring")
execs = job.get_executions()
ex = sorted(execs, key=lambda e: e.id)[-1]
print("execution", ex.id, ex.state, ex.final_status)
out, err = ex.download_logs()
for name in (out, err):
    print("=" * 20, name)
    with open(name) as f:
        print(f.read()[-8000:])
