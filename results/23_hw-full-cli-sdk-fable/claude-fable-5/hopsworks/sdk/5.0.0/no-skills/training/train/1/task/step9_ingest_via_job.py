import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()
js = project.get_job_api()

wrapper = '''import os
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
'''
with open("job_wrapper.py", "w") as fh:
    fh.write(wrapper)
print("uploaded:", ds.upload("job_wrapper.py", "Resources/trainjob646af0", overwrite=True))
assert ds.exists("Resources/trainjob646af0/job_wrapper.py")

job = js.get_job("trainjob646af0")
execution = job.run(await_termination=True)
print("final state:", execution.state, "| success:", execution.success)
