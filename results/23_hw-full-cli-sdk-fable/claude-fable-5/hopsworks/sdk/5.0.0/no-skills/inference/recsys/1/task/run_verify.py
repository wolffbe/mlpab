import os

for var in ("NO_PROXY", "no_proxy"):
    os.environ.pop(var, None)

import hopsworks

project = hopsworks.login()
project.get_dataset_api().upload("verify_recs_job.py", "Resources", overwrite=True)
jobs_api = project.get_job_api()
cfg = jobs_api.get_configuration("PYTHON")
cfg["appPath"] = "/Projects/" + project.name + "/Resources/verify_recs_job.py"
job = jobs_api.create_job("recs48963e_verify", cfg)
execution = job.run(await_termination=True)
print("final state:", execution.state, execution.final_status)
out, err = execution.download_logs()
print(open(out).read())
print("---- stderr tail ----")
print(open(err).read()[-1500:])
