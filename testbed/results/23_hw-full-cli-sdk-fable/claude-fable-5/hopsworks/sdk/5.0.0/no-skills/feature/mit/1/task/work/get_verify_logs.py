import hopsworks

project = hopsworks.login()
job_api = project.get_job_api()
job = job_api.get_job("featureseb4964_verify")
ex = max(job.get_executions(), key=lambda e: e.id)
print("execution:", ex.id, ex.state, ex.final_status)
ex.download_logs(path="work/logs")
