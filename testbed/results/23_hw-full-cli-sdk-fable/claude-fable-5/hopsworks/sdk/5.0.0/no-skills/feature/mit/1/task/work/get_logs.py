import hopsworks

project = hopsworks.login()
job_api = project.get_job_api()
job = job_api.get_job("featureseb4964_build")
exs = job.get_executions()
print("executions:", [(e.id, e.state) for e in exs])
ex = max(exs, key=lambda e: e.id)
print("newest execution id:", ex.id, "state:", ex.state, "final:", ex.final_status)
ex.download_logs(path="work/logs")
print("downloaded")
