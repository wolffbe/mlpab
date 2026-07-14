import hopsworks

proj = hopsworks.login()
dataset_api = proj.get_dataset_api()
job_api = proj.get_job_api()

dataset_api.upload("data/prediction_log.csv", "Resources", overwrite=True)
dataset_api.upload(".tmp/remote_monitor_job.py", "Resources", overwrite=True)
print("uploaded inputs")

config = job_api.get_configuration("PYTHON")
config["appPath"] = f"/Projects/{proj.name}/Resources/remote_monitor_job.py"
job = job_api.create_job("prediction_monitoring", config)
print("job created:", job.name)

execution = job.run(await_termination=True)
print("execution state:", execution.state, execution.final_status)
