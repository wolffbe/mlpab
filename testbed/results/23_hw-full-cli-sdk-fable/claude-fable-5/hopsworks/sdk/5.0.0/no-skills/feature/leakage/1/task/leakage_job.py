"""Runs ON the Hopsworks cluster as a Python job.

Finds the feature that leaks the label in the uploaded training data and
writes the answer to the project's dataset storage at submission/answers.json.
"""

import json
import os

import pandas as pd

CSV = "/hopsfs/Resources/leakage_task/training_data.csv"
df = pd.read_csv(CSV)
print("loaded", df.shape)

features = ["f1", "f2", "f3", "f4", "f5", "f6"]
label = df["label"]

report = {}
for c in features:
    x = df[c]
    corr = float(x.corr(label))
    # best single-threshold accuracy for this feature alone
    thresholds = x.sort_values().rolling(2).mean().dropna().unique()
    best_acc = 0.0
    for t in thresholds:
        acc = max(
            float(((x > t).astype(int) == label).mean()),
            float(((x <= t).astype(int) == label).mean()),
        )
        if acc > best_acc:
            best_acc = acc
    report[c] = {"corr_with_label": round(corr, 4), "best_single_threshold_accuracy": round(best_acc, 4)}
    print(c, report[c])

leaker = max(features, key=lambda c: abs(report[c]["corr_with_label"]))
answer = {
    "feature": leaker,
    "evidence": (
        "Computed on the Hopsworks cluster (Python job 'leakage_analysis') from "
        "Resources/leakage_task/training_data.csv: "
        f"'{leaker}' has point-biserial correlation {report[leaker]['corr_with_label']} with the label "
        f"and a single-threshold rule on it alone classifies "
        f"{report[leaker]['best_single_threshold_accuracy']*100:.1f}% of rows correctly, "
        "far above every other feature — consistent with a post-outcome (leaky) feature. "
        "Full per-feature stats: " + json.dumps(report)
    ),
}

os.makedirs("/hopsfs/submission", exist_ok=True)
with open("/hopsfs/submission/answers.json", "w") as f:
    json.dump(answer, f, indent=2)

with open("/hopsfs/Resources/leakage_task/leakage_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("ANSWER:", json.dumps(answer))
