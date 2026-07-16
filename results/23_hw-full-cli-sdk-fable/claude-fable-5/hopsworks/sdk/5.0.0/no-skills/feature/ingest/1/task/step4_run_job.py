import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()
jobs_api = project.get_job_api()

if not ds.exists("Resources/transactions_ingest"):
    ds.mkdir("Resources/transactions_ingest")

up1 = ds.upload("data/transactions_export_1.csv", "Resources/transactions_ingest", overwrite=True)
up2 = ds.upload("data/transactions_export_2.csv", "Resources/transactions_ingest", overwrite=True)
script_path = ds.upload("ingest_job.py", "Resources/transactions_ingest", overwrite=True)
print("uploaded:", up1, up2, script_path)

cfg = jobs_api.get_configuration("PYTHON")
cfg["appPath"] = script_path
cfg["environmentName"] = "python-feature-pipeline"
cfg["resourceConfig"]["memory"] = 4096

job = jobs_api.create_job("ingest_transactions82e347", cfg)
print("job created:", job.name)

execution = job.run(await_termination=True)
print("execution id:", execution.id)
print("final status:", execution.final_status, "success:", execution.success)

out, err = execution.get_logs()
print("=== STDOUT ===")
print(out)
print("=== STDERR ===")
print(err)
