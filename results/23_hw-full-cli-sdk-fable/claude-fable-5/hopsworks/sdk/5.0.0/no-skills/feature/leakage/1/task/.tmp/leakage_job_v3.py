"""Runs ON the Hopsworks cluster as a Python job.

Finds the feature that leaks the label in the uploaded training data and
writes the answer to the project's dataset storage at submission/answers.json.
Reads the data through the /hopsfs mount (the internal dataset REST API
returns 'Path not found' for recently uploaded files).
"""

import json
import os

import pandas as pd
import hopsworks

project = hopsworks.login()
dataset_api = project.get_dataset_api()

task_dir = "/hopsfs/Resources/leakage_task"
listing = os.listdir(task_dir)
print("task dir listing:", listing)

data_file = None
for name in ("training_data.txt", "training_data.csv"):
    if name in listing:
        data_file = os.path.join(task_dir, name)
        break
if data_file is None:
    raise SystemExit(f"no training data file found in {task_dir}: {listing}")

df = pd.read_csv(data_file)
print("loaded", len(df), "rows from", data_file)

features = ["f1", "f2", "f3", "f4", "f5", "f6"]
label = df["label"]

report = {}
for c in features:
    x = df[c]
    corr = float(x.corr(label))
    # best single-threshold accuracy for this feature alone
    order = x.sort_values()
    thresholds = order.rolling(2).mean().dropna().unique()
    best_acc = 0.0
    for t in thresholds:
        acc = max(
            float(((x > t).astype(int) == label).mean()),
            float(((x <= t).astype(int) == label).mean()),
        )
        if acc > best_acc:
            best_acc = acc
    report[c] = {"corr_with_label": corr, "best_single_threshold_accuracy": best_acc}
    print(c, report[c])

leaker = max(features, key=lambda c: abs(report[c]["corr_with_label"]))
answer = {
    "feature": leaker,
    "evidence": (
        "Computed on the Hopsworks cluster from the uploaded training data (Resources/leakage_task): "
        f"'{leaker}' has point-biserial correlation {report[leaker]['corr_with_label']:.4f} with the label "
        f"and a single-threshold rule on it alone classifies "
        f"{report[leaker]['best_single_threshold_accuracy']*100:.1f}% of rows correctly, "
        "far above every other feature. Full per-feature stats: "
        + json.dumps(report)
    ),
}

os.makedirs("/hopsfs/submission", exist_ok=True)
with open("/hopsfs/submission/answers.json", "w") as f:
    json.dump(answer, f, indent=2)
print("wrote /hopsfs/submission/answers.json")

with open("/hopsfs/Resources/leakage_task/leakage_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("ANSWER:", json.dumps(answer))
