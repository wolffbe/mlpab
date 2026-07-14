import hopsworks

proj = hopsworks.login()
dataset_api = proj.get_dataset_api()
path = dataset_api.upload("data/prediction_log.csv", "Resources", overwrite=True)
print("uploaded to:", path)
print("exists:", dataset_api.exists("Resources/prediction_log.csv"))
print(dataset_api.list("Resources"))
