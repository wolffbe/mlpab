import os
import runpy

import hopsworks
import pandas as pd

os.chdir("/hopsfs/Resources/trainjob646af0")
runpy.run_path("train_model.py", run_name="__main__")
print("predictions.csv written:", os.path.exists("predictions.csv"))

df = pd.read_csv("predictions.csv")
project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_or_create_feature_group(
    name="predictions646af0",
    version=1,
    primary_key=["row_id"],
    online_enabled=True,
    description="Predictions from job trainjob646af0",
)
fg.insert(df, wait=True)
print("inserted rows:", len(df))
