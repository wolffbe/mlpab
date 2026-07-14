import hopsworks

project = hopsworks.login()
job_api = project.get_job_api()
job = job_api.get_job("ftjob6f5e78")
execution = job.run(await_termination=True)
print("final_status:", execution.final_status, "state:", execution.state)
print("success:", execution.success)
