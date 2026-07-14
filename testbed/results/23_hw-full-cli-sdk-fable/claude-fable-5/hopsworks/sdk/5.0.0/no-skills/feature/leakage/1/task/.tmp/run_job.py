import os

for var in ("NO_PROXY", "no_proxy"):
    os.environ.pop(var, None)

import hopsworks

project = hopsworks.login(hostname_verification=False)
jobs_api = project.get_job_api()

config = jobs_api.get_configuration("PYTHON")
config["appPath"] = f"/Projects/{project.name}/Resources/leak_analysis_job.py"

job = jobs_api.create_job("leak_analysis", config)
print("job created:", job.name)

execution = job.run(await_termination=True)
print("final state:", execution.final_status, execution.state)
