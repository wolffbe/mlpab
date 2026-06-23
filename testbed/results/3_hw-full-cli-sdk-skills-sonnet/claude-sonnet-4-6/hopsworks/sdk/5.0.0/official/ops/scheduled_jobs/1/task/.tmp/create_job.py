import hopsworks

project = hopsworks.login()
project_name = project.name
print(f"Connected to project: {project_name}")

# Get existing job (already created)
job_api = project.get_job_api()
job = job_api.get_job("heartbeat76bfab")
print(f"Got job: {job.name}")

# Schedule it hourly (Quartz cron: second minute hour day-of-month month day-of-week year)
schedule = job.schedule(
    cron_expression="0 0 * ? * * *",
)
print(f"Schedule set: {schedule}")

# Run one execution now and wait for it to complete
execution = job.run(await_termination=True)
print(f"Execution status: {execution.state}")
print(f"Done.")
