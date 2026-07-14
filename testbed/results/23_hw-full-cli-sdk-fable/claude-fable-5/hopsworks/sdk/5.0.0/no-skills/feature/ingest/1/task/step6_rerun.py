import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()
jobs_api = project.get_job_api()

script_path = ds.upload("ingest_job.py", "Resources/transactions_ingest", overwrite=True)
print("uploaded:", script_path)

job = jobs_api.get_job("ingest_transactions82e347")
try:
    execution = job.run(await_termination=True)
finally:
    execs = job.get_executions()
    ex = sorted(execs, key=lambda e: e.id)[-1]
    print("execution:", ex.id, ex.state, ex.final_status)
    ex.download_logs()
