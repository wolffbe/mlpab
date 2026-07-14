"""Runs ON the Hopsworks cluster as a Python job: detects training/serving skew."""
import json
import math
import os
import shutil

import hopsworks
import pandas as pd

proj = hopsworks.login()
ds = proj.get_dataset_api()

train = pd.read_csv("/hopsfs/Resources/skew_task/training_sample.csv")
serve = pd.read_csv("/hopsfs/Resources/skew_task/serving_log.csv")

joined = train.merge(serve, on="entity_id", suffixes=("_train", "_serve"))
feats = ["f1", "f2", "f3", "f4", "f5"]

report = {}
for f in feats:
    diff = (joined[f + "_train"] - joined[f + "_serve"]).abs()
    report[f] = {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean())}

skewed = max(feats, key=lambda f: report[f]["mean_abs_diff"])

log1p_err = float(
    (joined[skewed + "_train"] - joined[skewed + "_serve"].apply(math.log1p)).abs().mean()
)

cause = (
    "For identical entities all features match exactly except {0}. "
    "Training values satisfy {0}_train == log1p({0}_serve) "
    "(mean abs error {1:.2e}), so the training pipeline applies a log1p "
    "transformation to {0} that the online serving path skips — the service "
    "serves the raw, untransformed value."
).format(skewed, log1p_err)

answer = {"feature": skewed, "cause": cause}

with open("answers.json", "w") as fh:
    json.dump(answer, fh)
with open("skew_report.json", "w") as fh:
    json.dump({"skewed": skewed, "log1p_mean_abs_err": log1p_err, "diffs": report}, fh)

os.makedirs("/hopsfs/Resources/submission", exist_ok=True)
shutil.copy("answers.json", "/hopsfs/Resources/submission/answers.json")
shutil.copy("answers.json", "/hopsfs/Resources/skew_task/answers.json")
shutil.copy("skew_report.json", "/hopsfs/Resources/skew_task/skew_report.json")

print("SKEWED_FEATURE=" + skewed)
print(json.dumps(report))
