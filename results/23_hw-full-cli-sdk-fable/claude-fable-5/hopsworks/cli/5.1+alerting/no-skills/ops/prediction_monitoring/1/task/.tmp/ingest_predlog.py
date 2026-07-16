"""Ingest deployed-model prediction log into the pred_log feature group and
compute statistics — runs as a Hopsworks PYTHON job (cluster-side)."""
import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()
local_path = dataset_api.download("Resources/prediction_log.csv", overwrite=True)

df = pd.read_csv(local_path)
df["ts"] = pd.to_datetime(df["ts"])
print(f"Loaded {len(df)} rows, {df['ts'].min()} .. {df['ts'].max()}")

fg = fs.get_feature_group("pred_log", version=1)
fg.insert(df, write_options={"wait_for_job": True})
print("Insert done")

fg.compute_statistics()
print("Statistics computed")
