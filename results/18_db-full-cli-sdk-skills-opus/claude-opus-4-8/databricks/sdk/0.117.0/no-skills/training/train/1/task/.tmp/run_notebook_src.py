# Databricks notebook source
import os, shutil, subprocess, sys, tempfile

VOL = "/Volumes/workspace/mlpab2138eb/trainjoba834e5_data"
work = tempfile.mkdtemp(prefix="trainjob_")
for fn in ["train.csv", "score.csv", "train_model.py"]:
    shutil.copy(f"{VOL}/{fn}", os.path.join(work, fn))

r = subprocess.run([sys.executable, "train_model.py"], cwd=work,
                   capture_output=True, text=True)
print("STDOUT:", r.stdout)
print("STDERR:", r.stderr)
r.check_returncode()

src = os.path.join(work, "predictions.csv")
shutil.copy(src, f"{VOL}/predictions.csv")
print("DONE predictions bytes:", os.path.getsize(f"{VOL}/predictions.csv"))
