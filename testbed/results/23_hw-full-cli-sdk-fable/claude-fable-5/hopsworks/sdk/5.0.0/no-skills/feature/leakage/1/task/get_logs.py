import hopsworks

project = hopsworks.login()
jobs_api = project.get_job_api()
job = jobs_api.get_job("leakage_analysis")
execution = job.get_executions()[0]
print("state:", execution.state, execution.final_status, "id:", execution.id)
out, err = execution.download_logs()
print("=== stdout ===")
print(open(out).read())
print("=== stderr ===")
print(open(err).read()[-6000:])
