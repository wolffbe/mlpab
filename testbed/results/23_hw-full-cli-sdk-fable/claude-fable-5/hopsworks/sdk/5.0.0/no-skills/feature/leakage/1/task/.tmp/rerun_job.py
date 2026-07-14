import os

for var in ("NO_PROXY", "no_proxy"):
    os.environ.pop(var, None)

import hopsworks

project = hopsworks.login(hostname_verification=False)
ds = project.get_dataset_api()
p = ds.upload(".tmp/leak_analysis_job.py", "Resources", overwrite=True)
print("uploaded:", p)

jobs_api = project.get_job_api()
job = jobs_api.get_job("leak_analysis")
execution = job.run(await_termination=True)
print("final state:", execution.final_status, execution.state)
