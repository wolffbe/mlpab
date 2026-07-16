"""Job runner: fetches inputs, runs the provided finetune_model.py as-is,
and uploads finetuned_model.npz + metrics.json back to the project dataset."""
import os
import runpy

import hopsworks

BASE = "Resources/ftjob6f5e78"

project = hopsworks.login()
ds = project.get_dataset_api()

workdir = os.path.join(os.getcwd(), "ftwork")
os.makedirs(workdir, exist_ok=True)
for name in ["base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"]:
    ds.download(BASE + "/" + name, workdir, overwrite=True)

os.chdir(workdir)
runpy.run_path("finetune_model.py", run_name="__main__")

out = BASE + "/output"
if not ds.path_exists(out):
    ds.mkdir(out)
ds.upload("finetuned_model.npz", out, overwrite=True)
ds.upload("metrics.json", out, overwrite=True)
print("done:", open("metrics.json").read())
