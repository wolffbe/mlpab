import hopsworks

project = hopsworks.login()
dataset_api = project.get_dataset_api()
for name in [
    "Resources/feature_history.csv",
    "Resources/feature_history_4fa858.csv",
    "Resources/model.json",
    "Resources/model_4fa858.json",
    "Resources/scoring_job.py",
]:
    print(name, dataset_api.exists(name))
