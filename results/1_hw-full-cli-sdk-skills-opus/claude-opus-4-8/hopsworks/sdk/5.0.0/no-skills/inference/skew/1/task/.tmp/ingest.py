import hopsworks
import pandas as pd

proj = hopsworks.login()
fs = proj.get_feature_store()

train = pd.read_csv("data/training_sample.csv")
serve = pd.read_csv("data/serving_log.csv")
print("train", train.shape, "serve", serve.shape)

fg_train = fs.get_or_create_feature_group(
    name="skew_train", version=1, primary_key=["entity_id"],
    description="training feature matrix", online_enabled=False)
fg_train.insert(train, write_options={"wait_for_job": True})

fg_serve = fs.get_or_create_feature_group(
    name="skew_serve", version=1, primary_key=["entity_id"],
    description="serving feature log", online_enabled=False)
fg_serve.insert(serve, write_options={"wait_for_job": True})
print("INSERTED OK")
