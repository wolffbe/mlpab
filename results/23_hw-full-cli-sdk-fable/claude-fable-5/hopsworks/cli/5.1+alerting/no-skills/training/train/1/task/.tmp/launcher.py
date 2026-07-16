"""Job launcher: stage inputs from HopsFS, run the provided train_model.py
unmodified, and upload predictions.csv back to HopsFS."""
import runpy

import hopsworks

BASE = "Resources/trainjob646af0"

project = hopsworks.login()
dataset_api = project.get_dataset_api()

for name in ["train.csv", "score.csv", "train_model.py"]:
    dataset_api.download(f"{BASE}/{name}", overwrite=True)

runpy.run_path("train_model.py", run_name="__main__")

dataset_api.upload("predictions.csv", BASE, overwrite=True)
print("done: predictions.csv uploaded to", BASE)
