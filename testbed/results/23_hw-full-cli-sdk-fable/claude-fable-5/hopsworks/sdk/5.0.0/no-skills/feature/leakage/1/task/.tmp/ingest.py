import os

for var in ("NO_PROXY", "no_proxy"):
    os.environ.pop(var, None)

import hopsworks
import pandas as pd  # only to load the CSV for SDK ingestion

project = hopsworks.login(hostname_verification=False)
fs = project.get_feature_store()

df = pd.read_csv("data/training_data.csv")
print(df.dtypes)
print("rows:", len(df))

fg = fs.get_or_create_feature_group(
    name="leakage_training_data",
    version=1,
    primary_key=["row_id"],
    description="Training data for leakage detection",
    statistics_config={"enabled": True, "correlations": True, "histograms": True},
)
fg.insert(df, wait=True)
print("inserted feature group:", fg.name, fg.version)
