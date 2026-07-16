import os
os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)
import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()
p = ds.upload("standardize_job.py", "Resources/scaledaff2b3", overwrite=True)
print("uploaded", p)

jobs = project.get_job_api()
job = jobs.get_job("scaledaff2b3_ingest")
execution = job.run(await_termination=True)
print("final state:", execution.state, execution.final_status)
