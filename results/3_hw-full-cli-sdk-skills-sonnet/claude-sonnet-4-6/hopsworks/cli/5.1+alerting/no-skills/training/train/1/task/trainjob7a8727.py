"""Job wrapper for trainjob7a8727.

Copies train.csv and score.csv from HopsFS FUSE mount, runs the provided
deterministic logistic regression script, then uploads predictions.csv.
"""
import os
import shutil
import importlib.util

import hopsworks

# HopsFS is mounted at /hopsfs in the job environment
hopsfs_job_dir = "/hopsfs/Resources/jobs/trainjob7a8727"
cwd = os.getcwd()

shutil.copy(os.path.join(hopsfs_job_dir, "train.csv"), os.path.join(cwd, "train.csv"))
shutil.copy(os.path.join(hopsfs_job_dir, "score.csv"), os.path.join(cwd, "score.csv"))

script_path = os.path.join(hopsfs_job_dir, "train_model.py")
spec = importlib.util.spec_from_file_location("train_model", script_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.main()

project = hopsworks.login()
dataset_api = project.get_dataset_api()
dataset_api.upload(os.path.join(cwd, "predictions.csv"), "Resources/jobs/trainjob7a8727", overwrite=True)
print("Done — predictions.csv uploaded to HopsFS.")
