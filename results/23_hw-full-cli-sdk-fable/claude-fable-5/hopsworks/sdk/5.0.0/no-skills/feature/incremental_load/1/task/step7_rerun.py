import os

os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)

import hopsworks

project = hopsworks.login()
jobs_api = project.get_jobs_api()
ds_api = project.get_dataset_api()

script_path = ds_api.upload("incrementaljob76da9e.py", "Resources", overwrite=True)
print("script re-uploaded:", script_path)

job = jobs_api.get_job("incrementaljob76da9e")
execution = job.run(await_termination=True)
print("execution state:", execution.final_status, "success:", execution.success)

schedule = job.schedule(cron_expression="0 0 6 * * ?")
print("schedule attached:", schedule.cron_expression, "next:", schedule.next_execution_date_time)
