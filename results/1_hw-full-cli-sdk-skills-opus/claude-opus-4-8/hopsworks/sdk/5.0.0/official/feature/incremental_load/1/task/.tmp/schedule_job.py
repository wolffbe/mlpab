import inspect
import hopsworks

project = hopsworks.login()
api = project.get_job_api()
job = api.get_job("incrementaljob614551")

print("schedule signature:", inspect.signature(job.schedule))
print(job.schedule.__doc__)
