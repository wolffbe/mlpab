import hopsworks

project = hopsworks.login()
dataset_api = project.get_dataset_api()
dataset_api.upload("scoring_job.py", "Resources", overwrite=True)
print("script re-uploaded")

jobs_api = project.get_jobs_api()
job = jobs_api.get_job("scoring4fa858")
execution = job.run(await_termination=True)
print("execution state:", execution.state, "final status:", execution.final_status)
print("success:", execution.success)
