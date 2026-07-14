import hopsworks

project = hopsworks.login()
dataset_api = project.get_dataset_api()
job_api = project.get_job_api()

folder = "Resources/featureseb4964"
script_path = dataset_api.upload("work/verify_job.py", folder, overwrite=True)
assert dataset_api.exists(folder + "/verify_job.py"), "upload did not persist"
print("uploaded script to:", script_path)

cfg = job_api.get_configuration("PYTHON")
cfg["appPath"] = script_path
cfg["environmentName"] = "python-feature-pipeline"

job = job_api.create_job("featureseb4964_verify", cfg)
execution = job.run(await_termination=True)
print("final status:", execution.final_status, "| state:", execution.state)
