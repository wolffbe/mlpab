"""Wrapper for trainjob7a8727 — copies data to /tmp, runs train_model, uploads output."""
import os
import sys
import shutil

script_dir = os.path.dirname(os.path.abspath(__file__))
print("Script dir:", script_dir)

# Copy data and script to /tmp (guaranteed writable)
work_dir = "/tmp/trainjob7a8727"
os.makedirs(work_dir, exist_ok=True)
for fname in ["train.csv", "score.csv", "train_model.py"]:
    src = os.path.join(script_dir, fname)
    dst = os.path.join(work_dir, fname)
    shutil.copy2(src, dst)
    print(f"Copied {fname} to {work_dir}")

os.chdir(work_dir)
sys.path.insert(0, work_dir)

import train_model
train_model.main()

output_path = os.path.join(work_dir, "predictions.csv")
print("predictions.csv exists:", os.path.exists(output_path))
if os.path.exists(output_path):
    with open(output_path) as f:
        print("First lines:", f.read()[:200])

import hopsworks
project = hopsworks.login()
dataset_api = project.get_dataset_api()
dataset_api.upload(output_path, "Resources/trainjob7a8727", overwrite=True)
print("Done — predictions.csv uploaded.")
