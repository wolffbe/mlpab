import os

os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)

import glob

import hopsworks

project = hopsworks.login()
jobs_api = project.get_jobs_api()
ds_api = project.get_dataset_api()

DATA_DIR = "Resources/incremental76da9e"
if not ds_api.exists(DATA_DIR):
    ds_api.mkdir(DATA_DIR)

for f in sorted(glob.glob("data/increment_*.csv")):
    p = ds_api.upload(f, DATA_DIR, overwrite=True)
    print("uploaded:", p)

script_path = ds_api.upload("incrementaljob76da9e.py", "Resources", overwrite=True)
print("script uploaded:", script_path)

config = jobs_api.get_configuration("PYTHON")
config["appPath"] = script_path
job = jobs_api.create_job("incrementaljob76da9e", config)
print("job created:", job.name, job.job_type)

execution = job.run(await_termination=True)
print("execution state:", execution.final_status, execution.success)

schedule = job.schedule(cron_expression="0 0 6 * * ?")
print("schedule attached:", schedule.cron_expression, "next:", schedule.next_execution_date_time)
