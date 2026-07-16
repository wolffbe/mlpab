import hopsworks

project = hopsworks.login()
jobs_api = project.get_jobs_api()
job = jobs_api.get_job("scoring4fa858")
execution = job.get_executions()[0]
print("exec id:", execution.id, "state:", execution.state)
out_path, err_path = execution.download_logs()
print("OUT LOG ----")
with open(out_path) as f:
    print(f.read()[-6000:])
print("ERR LOG ----")
with open(err_path) as f:
    print(f.read()[-6000:])
