import os

for var in ("NO_PROXY", "no_proxy"):
    os.environ.pop(var, None)

import hopsworks

project = hopsworks.login(hostname_verification=False)
jobs_api = project.get_job_api()

config = jobs_api.get_configuration("PYTHON")
config["appPath"] = f"/Projects/{project.name}/Resources/leakage_task/leakage_job.py"

job = jobs_api.create_job("leakage_detection", config)
execution = job.run(await_termination=True)
print("final state:", execution.final_status, execution.state)

out, err = execution.download_logs()
with open(out) as fh:
    print("STDOUT:\n", fh.read())
with open(err) as fh:
    print("STDERR (tail):\n", fh.read()[-3000:])
