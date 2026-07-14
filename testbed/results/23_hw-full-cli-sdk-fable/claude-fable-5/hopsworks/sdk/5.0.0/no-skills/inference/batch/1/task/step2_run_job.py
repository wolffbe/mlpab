import hopsworks

project = hopsworks.login()
dataset_api = project.get_dataset_api()

dataset_api.upload("data/feature_history.csv", "Resources", overwrite=True)
# give them task-specific names on the platform
if dataset_api.exists("Resources/feature_history_4fa858.csv"):
    dataset_api.remove("Resources/feature_history_4fa858.csv")
dataset_api.move("Resources/feature_history.csv", "Resources/feature_history_4fa858.csv")

dataset_api.upload("data/model.json", "Resources", overwrite=True)
if dataset_api.exists("Resources/model_4fa858.json"):
    dataset_api.remove("Resources/model_4fa858.json")
dataset_api.move("Resources/model.json", "Resources/model_4fa858.json")

dataset_api.upload("scoring_job.py", "Resources", overwrite=True)
print("uploads done")

jobs_api = project.get_jobs_api()
config = jobs_api.get_configuration("PYTHON")
config["appPath"] = "/Projects/{}/Resources/scoring_job.py".format(project.name)
job = jobs_api.create_job("scoring4fa858", config)
print("job created:", job.name)

execution = job.run(await_termination=True)
print("execution state:", execution.state, "final status:", execution.final_status)
print("success:", execution.success)
