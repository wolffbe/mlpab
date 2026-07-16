import os

os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)

import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
jobs_api = project.get_jobs_api()

job = jobs_api.get_job("incrementaljob76da9e")
print("job:", job.name, "| type:", job.job_type)
print("job_schedule:", job.job_schedule)

execution = job.get_executions()[0]
print("latest execution:", execution.id, execution.final_status)
out, err = execution.download_logs()
print("=== job STDOUT tail ===")
print(open(out).read()[-1500:])

fg = fs.get_feature_group("incremental76da9e", version=1)
print("fg:", fg.name, "v", fg.version)
print("online_enabled:", fg.online_enabled)
print("primary_key:", fg.primary_key)
print("event_time:", fg.event_time)
try:
    details = fg.commit_details()
    for cid, d in details.items():
        print("commit", cid, d)
except Exception as e:
    print("commit_details failed:", e)
