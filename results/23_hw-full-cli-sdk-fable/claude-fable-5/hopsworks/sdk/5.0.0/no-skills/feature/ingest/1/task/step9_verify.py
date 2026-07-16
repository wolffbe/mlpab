import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()
jobs_api = project.get_job_api()

script_path = ds.upload("verify_job.py", "Resources/transactions_ingest", overwrite=True)
print("uploaded:", script_path)
print("exists:", ds.exists("Resources/transactions_ingest/verify_job.py"))

cfg = jobs_api.get_configuration("PYTHON")
cfg["appPath"] = script_path
cfg["environmentName"] = "python-feature-pipeline"

job = jobs_api.create_job("verify_transactions82e347", cfg)
try:
    execution = job.run(await_termination=True)
finally:
    execs = job.get_executions()
    ex = sorted(execs, key=lambda e: e.id)[-1]
    print("execution:", ex.id, ex.state, ex.final_status)
    ex.download_logs()
