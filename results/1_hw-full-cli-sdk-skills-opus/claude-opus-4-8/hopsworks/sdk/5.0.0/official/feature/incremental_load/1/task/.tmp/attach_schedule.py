import hopsworks

project = hopsworks.login()
api = project.get_job_api()
job = api.get_job("incrementaljob614551")

# Daily at 02:00 UTC (Quartz cron: sec min hour day-of-month month day-of-week)
sched = job.schedule("0 0 2 ? * *")
print("Schedule attached:", sched)

# Re-fetch to confirm persistence
job2 = api.get_job("incrementaljob614551")
js = job2.job_schedule
print("Persisted job_schedule:", js)
print("cron_expression:", getattr(js, "cron_expression", None))
print("enabled:", getattr(js, "enabled", None))
