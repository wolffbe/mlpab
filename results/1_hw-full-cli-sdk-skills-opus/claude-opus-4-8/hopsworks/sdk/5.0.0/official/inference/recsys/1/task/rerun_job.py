import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()
job_dir = "Resources/jobs/recsfd473b"
print("re-upload script ->", ds.upload("recs_job.py", job_dir, overwrite=True))

api = project.get_job_api()
job = api.get_job("recsfd473b_job")
execution = job.run(await_termination=True)
print("execution final_status", execution.final_status, "success", execution.success)
