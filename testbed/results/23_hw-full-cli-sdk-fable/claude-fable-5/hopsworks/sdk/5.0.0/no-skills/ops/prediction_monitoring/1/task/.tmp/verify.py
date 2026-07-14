import hopsworks

proj = hopsworks.login()
dataset_api = proj.get_dataset_api()
print("answers.json exists:", dataset_api.exists("Resources/answers.json"))
print("content:", dataset_api.read_content("/Resources/answers.json").content.decode())

fs = proj.get_feature_store()
fg = fs.get_feature_group("prediction_log", 1)
print("feature group:", fg.name, "v", fg.version, "features:", [f.name for f in fg.features])
