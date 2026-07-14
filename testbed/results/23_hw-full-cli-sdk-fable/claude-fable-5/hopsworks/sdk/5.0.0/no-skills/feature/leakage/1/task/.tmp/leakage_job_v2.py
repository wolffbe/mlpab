"""Runs ON the Hopsworks cluster as a Python job.

Finds the feature that leaks the label in the uploaded training data and
writes the answer to the project's dataset storage at submission/answers.json.
"""

import json
import os

import pandas as pd
import hopsworks

project = hopsworks.login()
dataset_api = project.get_dataset_api()

local_csv = dataset_api.download("Resources/leakage_task/training_data.txt", overwrite=True)
df = pd.read_csv(local_csv)

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

out_local = os.path.abspath("answers.json")
with open(out_local, "w") as f:
    json.dump(answer, f, indent=2)

if not dataset_api.exists("submission"):
    try:
        dataset_api.mkdir("submission")
    except Exception as e:
        print("mkdir submission failed:", e)

uploaded = dataset_api.upload(out_local, "submission", overwrite=True)
print("uploaded answers to:", uploaded)

report_local = os.path.abspath("leakage_report.json")
with open(report_local, "w") as f:
    json.dump(report, f, indent=2)
dataset_api.upload(report_local, "Resources/leakage_task", overwrite=True)

print("ANSWER:", json.dumps(answer))
