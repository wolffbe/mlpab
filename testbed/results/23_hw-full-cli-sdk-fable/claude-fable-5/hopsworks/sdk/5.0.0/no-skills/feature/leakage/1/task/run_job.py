"""Driver: uploads data + job script, runs the analysis as a Hopsworks job."""

import hopsworks

project = hopsworks.login()
dataset_api = project.get_dataset_api()

if not dataset_api.exists("Resources/leakage_task"):
    dataset_api.mkdir("Resources/leakage_task")

csv_path = dataset_api.upload("data/training_data.csv", "Resources/leakage_task", overwrite=True)
script_path = dataset_api.upload("leakage_job.py", "Resources/leakage_task", overwrite=True)
print("uploaded:", csv_path, script_path)

jobs_api = project.get_job_api()
config = jobs_api.get_configuration("PYTHON")
config["appPath"] = "/Projects/" + project.name + "/Resources/leakage_task/leakage_job.py"
job = jobs_api.create_job("leakage_analysis", config)
print("job created:", job.name)

execution = job.run(await_termination=True)
print("final status:", execution.final_status, "state:", execution.state)
out, err = execution.download_logs()
print("=== stdout ===")
print(open(out).read())
print("=== stderr (tail) ===")
print(open(err).read()[-4000:])
