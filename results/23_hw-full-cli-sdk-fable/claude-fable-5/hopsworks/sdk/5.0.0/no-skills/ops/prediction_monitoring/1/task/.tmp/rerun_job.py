import hopsworks

proj = hopsworks.login()
dataset_api = proj.get_dataset_api()
job_api = proj.get_job_api()

dataset_api.upload(".tmp/remote_monitor_job.py", "Resources", overwrite=True)
print("uploaded script")

job = job_api.get_job("prediction_monitoring")
execution = job.run(await_termination=True)
print("execution state:", execution.state, execution.final_status)
out, err = execution.download_logs()
for name in (out, err):
    print("=" * 20, name)
    with open(name) as f:
        print(f.read()[-6000:])
