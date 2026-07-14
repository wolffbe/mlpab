import hopsworks

project = hopsworks.login()
dataset_api = project.get_dataset_api()
job_api = project.get_job_api()

folder = "Resources/featureseb4964"
if not dataset_api.exists(folder):
    dataset_api.mkdir(folder)

dataset_api.upload("data/transactions.csv", folder, overwrite=True)
dataset_api.upload("data/fx_rates.csv", folder, overwrite=True)
script_path = dataset_api.upload("work/fg_job.py", folder, overwrite=True)
print("uploaded script to:", script_path)

cfg = job_api.get_configuration("PYTHON")
cfg["appPath"] = script_path
cfg["environmentName"] = "python-feature-pipeline"
cfg["resourceConfig"]["memory"] = 4096
cfg["resourceConfig"]["cores"] = 2.0

job = job_api.create_job("featureseb4964_build", cfg)
print("job created:", job.name)

execution = job.run(await_termination=True)
print("final status:", execution.final_status, "| state:", execution.state)

out, err = execution.get_output()
print("=== STDOUT ===")
print(out)
print("=== STDERR ===")
print(err)
